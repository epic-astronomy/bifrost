#!/usr/bin/env python

# Copyright (c) 2019, The Bifrost Authors. All rights reserved.
# Copyright (c) 2019, The University of New Mexico. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of The Bifrost Authors nor the names of its
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import print_function

import os
import sys
import glob
import warnings
import argparse
from textwrap import fill as tw_fill


# Python wrapper template
_WRAPPER_TEMPLATE = r"""
from bifrost.libbifrost import _check, _get, BifrostObject
from bifrost.ndarray import asarray

import {libname}_generated as _gen

"""


def _patch_bifrost_objects(filename, includes):
    """
    Given a wrapper generated by ctypesgen, clean up the file to make sure
    that it uses the "official" Bifrost objects.
    """
    
    # Deal with the includes by converting a string to a
    # list.
    if isinstance(includes, str):
        includes = includes.split(None)
    
    # Load it in a process it
    wrapper = []
    with open(filename, 'r') as fh:
        inside_ignore = False
        first_post_ignore = True
        for line in fh:
            # "# No modules" is the last valid line before we hit the internal Bifrost
            # definitions that we need to cut out
            if line.startswith("# No modules") or line.startswith("# End modules"):
                inside_ignore = line
                
            # If we are inside the internal Bifrost definitions, look for a sign that
            # we are done by finding one of our include files mentioned in a comment
            if inside_ignore:
                for inclue in includes:
                    if line[0] == '#' and line.find(inclue) != -1:
                        if first_post_ignore:
                            wrapper.append(inside_ignore)
                            wrapper.append("\nfrom bifrost.libbifrost_generated import *\n\n")
                            first_post_ignore = True
                        inside_ignore = False
                        break
            if not inside_ignore:
                wrapper.append(line)
                
    # A warning, if necessary
    if not first_post_ignore:
        warnings.warn(RuntimeWarning, "File may not have been wrapped correctly")
        
    # Write it back out
    with open(filename, 'w') as fh:
        fh.write("".join(wrapper))


def _normalize_function_name(function):
    """
    Given a C++ function name, convert it to a nicely formatted Python
    function name.  For example "SetPositions" becomes "set_positions".
    """
    
    name = function[0].lower()
    for l in function[1:]:
        if l.isupper():
            name += '_'
        name += l.lower()
    return name


def _reverse_normalize_function_name(function):
    """
    Given a Python funciton name, convert it into a C++ function name.  For
    example "set_positions" becomes "SetPositions".
    """
    
    name = function[0].upper()
    next_upper = False
    for l in function[1:]:
        if l == '_':
            next_upper = True
            continue
        name += l.upper() if next_upper else l
        next_upper = False
    return name

def _split_and_clean_args(args):
    """
    Given a string of arguments from a ctypesgen-generated wrapper, parse
    them into a list of something we can use later.
    """
    
    args = args.replace('[', '').replace(']', '')
    args = args.split(',')
    args = [arg.strip().rstrip() for arg in args]
    for i in range(len(args)):
        # Deal with pointers
        if args[i].startswith('POINTER'):
            if args[i].find('BFarray') != -1:
                args[i] = 'BFarray'
            elif args[i] == 'POINTER(None)':
                args[i] = 'ptr_generic'
            elif args[i] == 'POINTER(POINTER(None))':
                args[i] = 'ptr_ptr_generic'
            else:
                ctype = args[i].split('(', 1)[1]
                ctype = ctype.replace(')', '')
                args[i] = "ptr_%s" % ctype
    return args


def _extract_calls(filename, libname):
    """
    Given a wrapper generated by ctypesgen, extract all of the function call 
    information and return it as a dictionary.
    """
    
    # Load the file in
    wrapper = []
    with open(filename, 'r') as fh:
        for line in fh:
            wrapper.append(line)
            
    # Pass 1:  Find the functions and defintion locations
    functions = {}
    locations = {}
    for i,line in enumerate(wrapper):
        if line.find('if not hasattr(_lib') != -1 or line.find('if hasattr(_lib') != -1:
            function = line.split(None)[-1][1:]
            function = function.replace("'):", '')
            py_name = function.split(_reverse_normalize_function_name(libname), 1)[-1]
            if py_name == '':
                ## Catch for when the library name is the same as the function
                py_name = function
            py_name = _normalize_function_name(py_name)
            functions[py_name] = function
            if wrapper[i-1][0] == '#' and wrapper[i-1].find(':') != -1:
                locations[py_name] = wrapper[i-1].split(None, 1)[1]
            elif wrapper[i-2][0] == '#' and wrapper[i-2].find(':') != -1:
                locations[py_name] = wrapper[i-2].split(None, 1)[1]
            
    # Pass 2: Find the argument and return types
    arguments, results = {}, {}
    for line in wrapper:
        for py_name in functions:
            c_name = functions[py_name]
            if line.find("%s.argtypes" % c_name) != -1:
                value = line.split('=', 1)[-1]
                arguments[py_name] = _split_and_clean_args(value)
            elif line.find("%s.restype" % c_name) != -1:
                value = line.split('=', 1)[-1]
                results[py_name] = _split_and_clean_args(value)
                
    # Pass 3:  Find the argument names
    names = {}
    for py_name in locations:
        filename, line_no = locations[py_name].split(':',1)
        line_no = int(line_no, 10)
        
        definition = ''
        with open(filename, 'r') as fh:
            i = 0
            for line in fh:
                i += 1
                if i < line_no:
                    continue
                definition += line.strip().rstrip()
                if line.find(')') != -1:
                    break
        definition = definition.split('(', 1)[1]
        definition = definition.split(')', 1)[0]
        args = definition.split(',')
        args = [arg.rsplit(None, 1)[1].replace('*', '') for arg in args]
        names[py_name] = args
        
    # Combine and done
    calls = {}
    for py_name in functions:
        calls[py_name] = {}
        calls[py_name]['c_name'] = functions[py_name]
        calls[py_name]['arguments'] = arguments[py_name]
        try:
            calls[py_name]['names'] = names[py_name]
        except KeyError:
            ## Fallback in case we didn't find anything useful
            calls[py_name]['names'] = ['arg%i' for i in range(len(arguments[py_name]))]
        calls[py_name]['results'] = results[py_name]
    return calls


def _class_or_functions(calls):
    """
    Given a dictionary of call signatures extracted by _extract_calls(), 
    determine what kind of wrapper to build.  Options are 'class' for 
    something that looks like it shoud be a class or 'functions' for a 
    collection of functions.
    """
    
    wrap_type = "functions"
    if 'create' in calls \
       and 'destroy' in calls \
       and 'init' in calls \
       and 'execute'in calls:
        wrap_type = "class"
    return wrap_type


def _check_get_or_return(py_name, call):
    """
    Given a call signature extracted by _extract_calls(), figure out how
    to deal with the outcome of the function call.  Valid options are 
    'return' to just return it, 'check' to just call the Bifrost _check()
    function, or 'get' to call the Bifrost _get() function.
    """
    
    args = call['arguments']
    ress = call['results']
    
    if ress[0] != 'BFstatus':
        return 'return'
    else:
        if args[-1].startswith('ptr') and not py_name.startswith('set_'):
            return 'get'
        else:
            return 'check'


def _convert_call_args(call, for_method=True):
    """
    Convert a set of arguments associated with a call signature extracted by
    _extract_calls() into two strings:  one for Python calls and one for 
    wrapped C calls.
    """
    
    args = call['arguments']
    names = call['names']
    
    py_args, c_args = [], []
    for arg,name in zip(args, names):
        py_args.append("%s_%s" % (name, arg))
        c_args.append(py_args[-1])
        # Make sure we wrap arguments of type BFarray with 
        # "asarray().as_BFarray()".
        if c_args[-1].find('_BFarray') != -1:
            c_args[-1] = "asarray(%s).as_BFarray()" % c_args[-1]
    if for_method:
        py_args[0] = 'self'
        c_args[0] = 'self.obj'
        
    py_args = ', '.join(py_args)
    c_args = ', '.join(c_args)
    return py_args, c_args


def _render_call(py_name, call, for_method=False, indent=0):
    """
    Given a Python function name and its associated call signature, return
    a string corresponding to full Python function definition.
    """
    
    c_name = call['c_name']
    py_args, c_args = _convert_call_args(call, for_method=for_method)
    call_base = "_gen.{name}({args})".format(name=c_name, args=c_args)
    return_type = _check_get_or_return(py_name, call)
    
    tw_padding = indent + 4 + len(c_name) + 5 + 1
    if return_type == 'check':
        tw_padding += 7
    elif return_type == 'get':
        tw_padding += 5
    else:
        tw_padding += 7
    call_base = tw_fill(call_base,
                        subsequent_indent=' '*tw_padding,
                        break_long_words=False,
                        break_on_hyphens=False)
    
    output = ''
    output += "{indent}def {name}({args}):\n".format(indent=' '*indent,
                                                     name=py_name,
                                                     args=py_args)
    if return_type == 'check':
        output += "{indent}    _check({call})\n".format(indent=' '*indent,
                                                        call=call_base)
        if py_name == 'execute':
            py_arrays = [arg.strip() for arg in py_args.split(',') if arg.find('BFarray') != -1]
            if len(py_arrays) > 1:
                output += "{indent}    return {array}\n".format(indent=' '*indent,
                                                                array=py_arrays[-1])
    elif return_type == 'get':
        output += "{indent}    _get({call})\n".format(indent=' '*indent,
                                                      call=call_base)
    else:
        output += "{indent}    return {call}\n".format(indent=' '*indent,
                                                       call=call_base)
    output += '\n'
    
    return output


def main(args):
    filename = args.filename
    libname = os.path.basename(filename)
    libname = libname.split('_generated', 1)[0]
    wrapname = "%s.py" % libname
    
    includes = glob.glob("%s*.h" % libname)
    includes.extend(glob.glob("%s*.hpp" % libname))
    
    # Patch
    print("INFO: Status: patching %s." % os.path.basename(filename))
    _patch_bifrost_objects(filename, includes)
    
    # Extract the ctypes-wrapped calls and use that to find what type of 
    # wrapper to build
    print("INFO: Status: extracting function calls.")
    calls = _extract_calls(filename, libname)
    wrap_type = _class_or_functions(calls)
    
    # Build the wrapper
    print("INFO: Status: Writing to %s." % wrapname)
    with open(wrapname, 'w') as fh:
        template = _WRAPPER_TEMPLATE.format(libname=libname)
        fh.write(template)
        
        if wrap_type == 'functions':
            for py_name in calls:
                call = calls[py_name]
                function = _render_call(py_name, call, for_method=False, indent=0)
                fh.write(function)
        else:
            fh.write("class {wrapname}(BifrostObject):\n".format(wrapname=_reverse_normalize_function_name(libname)))
            fh.write("    def __init__(self):\n")
            fh.write("        BifrostObject.__init__(self, _gen.{create}, _gen.{destroy})\n".format(create=calls['create']['c_name'],
                                                                                                   destroy=calls['destroy']['c_name']))
            del calls['create']
            del calls['destroy']
            
            for py_name in ('init', 'execute'):
                call = calls[py_name]
                function = _render_call(py_name, call, for_method=True, indent=4)
                fh.write(function)
                del calls[py_name]
                
            ## everything else
            for py_name in calls:
                call = calls[py_name]
                function = _render_call(py_name, call, for_method=True, indent=4)
                fh.write(function)
    print("INFO: Status: High-level wrapping complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a high-level Python wrapper for a Bifrost plugin")
    parser.add_argument('filename', type=str,
                        help='ctypesgen module to parse')
    args = parser.parse_args()
    main(args)
    