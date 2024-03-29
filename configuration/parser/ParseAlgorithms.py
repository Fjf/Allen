#!/usr/bin/python3
###############################################################################
# (c) Copyright 2021 CERN for the benefit of the LHCb Collaboration           #
###############################################################################

import re
import os
import sys
import codecs
from collections import OrderedDict
from AlgorithmTraversalLibClang import AlgorithmTraversal
import argparse
import pickle
import json


def get_clang_so_location():
    """Function that fetches location of detected clang so."""
    import clang.cindex
    so_library = ""
    try:
        _ = clang.cindex.conf.lib.__test_undefined_symbol
    except AttributeError as error:
        so_library = error.args[0].split(":")[0]
    return so_library


class Parser():
    """A parser static class. This class steers the parsing of the
    codebase. It can be configured with the variables below."""

    # Pattern sought in every file, prior to parsing the file for an algorithm
    __algorithm_pattern_compiled = re.compile(
        "(?P<scope>Host|Device|Selection|Validation|Provider|Barrier)Algorithm"
    )

    # File extensions considered
    __sought_extensions_compiled = [
        re.compile(".*\\." + p + "$") for p in ["cuh", "h", "hpp"]
    ]

    # Folders storing device and host code
    __device_folder = "device"
    __host_folder = "host"

    @staticmethod
    def __get_filenames(folder, extensions):
        list_of_files = []
        for root, subdirs, files in os.walk(folder):
            for filename in files:
                for extension in extensions:
                    if extension.match(filename):
                        list_of_files.append(os.path.join(root, filename))
        return list_of_files

    @staticmethod
    def get_all_filenames(prefix_project_folder):
        return Parser.__get_filenames(prefix_project_folder + Parser.__device_folder, Parser.__sought_extensions_compiled) + \
            Parser.__get_filenames(
                prefix_project_folder + Parser.__host_folder, Parser.__sought_extensions_compiled)

    @staticmethod
    def find_algorithm_files(prefix_project_folder):
        all_filenames = Parser.get_all_filenames(prefix_project_folder)
        algorithm_files = []

        for filename in all_filenames:
            with codecs.open(filename, 'r', 'utf-8') as f:
                s = f.read()
                # Invoke the libTooling algorithm parser only if we find the algorithm pattern
                has_algorithm = Parser.__algorithm_pattern_compiled.search(s)
                if has_algorithm:
                    algorithm_files.append(filename)

        return algorithm_files

    @staticmethod
    def parse_all(algorithm_files,
                  prefix_project_folder,
                  algorithm_parser=AlgorithmTraversal()):
        """Parses all files and traverses algorithm definitions."""
        algorithms = []

        for algorithm_file in algorithm_files:
            try:
                parsed_algorithms = algorithm_parser.traverse(
                    algorithm_file, prefix_project_folder)
                if parsed_algorithms:
                    algorithms += parsed_algorithms
            except:
                print("Parsing file", algorithm_file, "failed")
                raise

        return algorithms


class AllenCore():
    """Static class that generates a python representation of
    Allen algorithms."""

    @staticmethod
    def prefix(indentation_level, indent_by=2):
        return "".join([" "] * indentation_level * indent_by)

    @staticmethod
    def create_var_type(kind):
        return ("R" if "input" in kind.lower() else "W")

    @staticmethod
    def write_preamble(i=0):
        # Fetch base_types.py and include it here to make file self-contained
        s = "\n".join([
            "from AllenCore.AllenKernel import AllenAlgorithm, AllenDataHandle",
            "from collections import OrderedDict\n\n"
        ])
        return s

    @staticmethod
    def write_algorithm_code(algorithm, default_properties, i=0):
        s = AllenCore.prefix(
            i) + "class " + algorithm.name + "(AllenAlgorithm):\n"
        i += 1

        # Slots
        s += AllenCore.prefix(i) + "__slots__ = OrderedDict(\n"
        i += 1
        for param in algorithm.parameters:
            dependencies = [
                "\"" + dep.replace(algorithm.namespace + "::Parameters::", "")
                + "\"" for dep in param.dependencies
            ]
            dependencies = "[" + \
                ", ".join(dependencies) + "]" if dependencies else "[]"
            s += AllenCore.prefix(i) + param.typename + " = AllenDataHandle(\"" + param.scope + "\", " + dependencies + ", \"" + param.typename + "\", \"" \
                + AllenCore.create_var_type(param.kind) + \
                "\", \"" + str(param.typedef) + "\"),\n"

        # Properties
        for prop in algorithm.properties:
            # Use the python JSON parser to turn the JSON
            # representation of default values into appropriate Python
            # objects
            pn = prop.name[1:-1]
            dv = json.loads(default_properties[pn])

            # Quotes have to be added for properties that hold a string
            if type(dv) is str:
                dv = f'"{dv}"'

            # Write the code for the property and include the C++ type
            # as a comment
            s += f'{AllenCore.prefix(i)}{pn} = {dv}, # {prop.typedef}\n'
        s = s[:-1]
        i -= 1
        s += "\n" + AllenCore.prefix(i) + ")\n"

        # aggregates: parameters marked optional or aggregate
        s += AllenCore.prefix(i) + "aggregates = ("
        i += 1
        for param in algorithm.parameters:
            if param.aggregate or param.optional:
                s += "\n" + AllenCore.prefix(i) + "\"" + param.typename + "\","
        i -= 1
        s += ")\n\n"

        s += AllenCore.prefix(i) + "@staticmethod\n"
        s += AllenCore.prefix(i) + "def category():\n"
        i += 1
        s += AllenCore.prefix(i) + f"return \"{algorithm.scope}\"\n\n"
        i -= 1

        s += AllenCore.prefix(i) + "def __new__(self, name, **kwargs):\n"
        i += 1
        s += AllenCore.prefix(
            i) + "instance = AllenAlgorithm.__new__(self, name)\n"
        s += AllenCore.prefix(i) + "for n,v in kwargs.items():\n"
        i += 1
        s += AllenCore.prefix(i) + "setattr(instance, n, v)\n"
        i -= 1
        s += AllenCore.prefix(i) + "return instance\n\n"
        i -= 1

        s += AllenCore.prefix(i) + "@classmethod\n"
        s += AllenCore.prefix(i) + "def namespace(cls):\n"
        i += 1
        s += AllenCore.prefix(i) + "return \"" + algorithm.namespace + "\"\n\n"
        i -= 1
        s += AllenCore.prefix(i) + "@classmethod\n"
        s += AllenCore.prefix(i) + "def filename(cls):\n"
        i += 1
        s += AllenCore.prefix(i) + "return \"" + algorithm.filename + "\"\n\n"
        i -= 1
        s += AllenCore.prefix(i) + "@classmethod\n"
        s += AllenCore.prefix(i) + "def getType(cls):\n"
        i += 1
        s += AllenCore.prefix(i) + "return \"" + algorithm.name + "\"\n\n\n"

        return s

    @staticmethod
    def generate_gaudi_wrapper_for_aggregate(algorithm):

        # Initialize the properties
        properties_initialization = [
            f"m_algorithm.set_property_value<{algorithm.namespace}::Parameters::{p.typename}, {p.typedef}>(m_{p.typename}.value());"
            if p.scope == "algorithm" else
            f"m_algorithm.set_property_value<Allen::Algorithm::{p.typename}, {p.typedef}>(m_{p.typename}.value());"
            for p in algorithm.properties
        ]
        properties = [
            f"Gaudi::Property<{p.typedef}> m_{p.typename}{{this, {p.name}, {p.default_value}, [=](auto&) {{ {init} }}, Gaudi::Details::Property::ImmediatelyInvokeHandler{{true}}, {p.description} }};"
            for p, init in zip(algorithm.properties, properties_initialization)
        ]
        properties += [
            # a property indicating that it contains optionals
            'Gaudi::Property<bool> m_hasOptionals{this, "hasOptionals", true};'
        ]

        # split into aggregates and normal inputs
        inputs = [
            parameter for parameter in algorithm.parameters
            if "input" in parameter.kind.lower() and not parameter.aggregate
        ]

        aggregates = [
            parameter for parameter in algorithm.parameters
            if "input" in parameter.kind.lower() and parameter.aggregate
        ]

        aggregate_types = [
            f"typename {algorithm.namespace}::Parameters::{agg.typename}::type::type"
            for agg in aggregates
        ]

        aggregate_handles = [
            f"std::vector<DataObjectReadHandle<Allen::parameter_vector<{typ}>>> m_{agg.typename};"
            for agg, typ in zip(aggregates, aggregate_types)
        ]
        aggregate_input_vectors = [
            "\n".join([
                f"#ifdef GAUDI_FUNCTIONAL_MAKE_VECTOR_OF_HANDLES_USES_DATAOBJID",
                f"Gaudi::Property<std::vector<DataObjID>> m_{agg.typename}_locations",
                f"#else",
                f"Gaudi::Property<std::vector<std::string>> m_{agg.typename}_locations",
                f"#endif",
                f"{{this, \"{agg.typename}\", {{}},",
                f"  [=]( Gaudi::Details::PropertyBase& ) {{",
                f"    this->m_{agg.typename} =",
                f"      Gaudi::Functional::details::make_vector_of_handles<decltype( this->m_{agg.typename} )>( this, m_{agg.typename}_locations );",
                f"    std::for_each( this->m_{agg.typename}.begin(), this->m_{agg.typename}.end(),",
                f"                    []( auto& h ) {{ h.setOptional( true ); }} );",
                f"}},",
                f"Gaudi::Details::Property::ImmediatelyInvokeHandler{{true}}}};",
            ]) for agg in aggregates
        ]

        input_types = [
            f"Allen::parameter_vector<{algorithm.namespace}::Parameters::{p.typename}::type>"
            for p in inputs
        ]

        input_handles = [
            f"DataObjectReadHandle<{typ}> m_{inp.typename} {{this, \"{inp.typename}\", \"\"}};"
            for inp, typ in zip(inputs, input_types)
        ] + [
            "DataObjectReadHandle<RuntimeOptions> m_runtime_options {this, \"runtime_options_t\", \"\"};",
            "DataObjectReadHandle<Constants const*> m_constants {this, \"constants_t\", \"\"};",
        ]

        outputs = [
            parameter for parameter in algorithm.parameters
            if "output" in parameter.kind.lower()
        ]

        output_types = [
            f"Allen::parameter_vector<{algorithm.namespace}::Parameters::{p.typename}::type>"
            for p in outputs
        ]
        output_handles = [
            f"DataObjectWriteHandle<{typ}> m_{out.typename} {{this, \"{out.typename}\", \"\"}};"
            for out, typ in zip(outputs, output_types)
        ]

        code = "\n".join((
            "#include \"AlgorithmConversionTools.h\"",
            f"#include <{algorithm.filename}>",
            "#include <Gaudi/Algorithm.h>",
            "#include <GaudiAlg/FunctionalDetails.h>",
            "#include <GaudiKernel/FunctionalFilterDecision.h>",
            "#include <vector>",
            "using namespace Gaudi::Functional;",
            f"class {algorithm.name} final : public Gaudi::Algorithm {{",
            "public:",
            "using Gaudi::Algorithm::Algorithm;",
            "StatusCode initialize() override {",
            "    const StatusCode sc = Algorithm::initialize();",
            "    if ( sc.isFailure() ) return sc;",
            "    Allen::initialize_algorithm(m_algorithm);",
            "    m_algorithm.set_name(this->name());",
            "    return sc;",
            "}",
            "",
            "private:",
            f"{algorithm.namespace}::{algorithm.name} m_algorithm{{}};\n",
        ))

        code += "\n" + "\n".join(input_handles + output_handles +
                                 aggregate_handles + aggregate_input_vectors)
        code += "\n" + "\n".join(properties)
        code += "\n" + "\n".join(
            ("public:",
             "StatusCode execute( const EventContext& ) const override {"))

        # loop over inputs to get them
        # required
        code += "\n".join(
            (f"auto const& {inp.typename} = *m_{inp.typename}.get();"
             for inp in inputs if not inp.optional))
        # optional
        code += "\n".join((
            f"auto const* {inp.typename}_ptr = m_{inp.typename}.getIfExists();\n"
            f"auto const& {inp.typename} = {inp.typename}_ptr ? *{inp.typename}_ptr : decltype(*{inp.typename}_ptr){{{{}},{{}}}};"
            for inp in inputs if inp.optional))
        # we need decltype(*{inp.typename}_ptr){{{{}},{{}}}} to initialize a vector with 2 elements (most often ints), such that
        # the typical access pattern of [event_number], [event_number + 1] for offsets works. Initializing explicitly with 0s fails
        # if more complicated types are to be initialized.
        # keep in mind that we only have single events in mind here, so that event_number == 0 is always true.
        # for gaudi this is a reasonable assumption as it only runs single events at a time.

        code += "\n"
        code += "auto const& runtime_options = *m_runtime_options.get();\n"
        code += "auto const& constants = *m_constants.get();\n"

        code += "\n".join((
            f"std::vector<{typ}> empty_vector_tes_wrappers_{agg.typename} {{}};\n"
            +
            f"std::vector<Allen::TESWrapperInput<{typ}>> tes_wrappers_{agg.typename};\n"
            +
            f"tes_wrappers_{agg.typename}.reserve(m_{agg.typename}.size());\n"
            + f"for (auto const& h : m_{agg.typename}) {{\n" +
            f"  auto* inp = h.getIfExists(); \n" +
            f"  tes_wrappers_{agg.typename}.emplace_back(inp ? *inp : empty_vector_tes_wrappers_{agg.typename}, \"{agg.typename}\");\n"
            + f"}}\n" +
            f"std::vector<std::reference_wrapper<Allen::Store::BaseArgument>> arg_data_{agg.typename};\n"
            + f"arg_data_{agg.typename}.reserve(m_{agg.typename}.size());\n" +
            f"for (auto& w : tes_wrappers_{agg.typename}) {{\n" +
            f"  arg_data_{agg.typename}.emplace_back(w);\n" + f"}}\n"
            for agg, typ in zip(aggregates, aggregate_types)))

        aggregate_types = [
            f"{algorithm.namespace}::Parameters::{agg.typename}::type"
            for agg in aggregates
        ]
        arg_data_agg_typenames = [
            f"arg_data_{agg.typename}" for agg in aggregates
        ]
        code += "std::tuple<" + ",".join(aggregate_types) + \
            "> input_aggregates_tuple {" + \
                ",".join(arg_data_agg_typenames) + "};"

        tes_wrappers_list = []
        tes_wrappers_reference_initialization_list = []
        output_container_element = 0

        parameters_non_aggregate = [
            p for p in algorithm.parameters if not p.aggregate
        ]

        for i, p in enumerate(parameters_non_aggregate):
            # Fetch the type of the TES wrapper for parameter p
            # Produce the initialization line for parameter p into a TESWrapper
            if p in inputs:
                tes_wrapper_type_text = "TESWrapperInput"
                parameter_variable_name = f"{p.typename}"
            else:
                tes_wrapper_type_text = "TESWrapperOutput"
                parameter_variable_name = f"std::get<{output_container_element}>(output_container)"
                output_container_element += 1

            tes_wrapper_initialization = f"{{{parameter_variable_name},\"{p.typename}\"}}"
            tes_wrapper_variable_name = f"{p.typename}_wrapper"

            tes_wrappers_list.append(
                f"Allen::{tes_wrapper_type_text}<{algorithm.namespace}::Parameters::{p.typename}::type> {tes_wrapper_variable_name} {tes_wrapper_initialization};"
            )
            tes_wrappers_reference_initialization_list.append(
                f"{tes_wrapper_variable_name}")

        tes_wrappers = "\n".join(tes_wrappers_list)
        tes_wrappers_reference_initialization = ",".join(
            tes_wrappers_reference_initialization_list)
        tes_wrappers_reference = f"std::array<std::reference_wrapper<Allen::Store::BaseArgument>, {len(parameters_non_aggregate)}> tes_wrappers_references {{{tes_wrappers_reference_initialization}}};"

        # lets call m_algorithm with our newly defined inputs
        # make teswrappers
        code += "\n".join((
            "// Output container",
            "std::tuple<" + ",".join(output_types) + "> output_container {};",
            "// TES wrappers", f"{tes_wrappers}",
            "// Inputs to set_arguments_size and operator()",
            f"{tes_wrappers_reference}", f"Allen::Context context{{}};",
            f"const auto argument_references = ArgumentReferences<{algorithm.namespace}::Parameters>{{tes_wrappers_references, input_aggregates_tuple}};",
            f"// set arguments size invocation",
            f"m_algorithm.set_arguments_size(argument_references, runtime_options, *constants);",
            f"// algorithm operator() invocation",
            f"m_algorithm(argument_references, runtime_options, *constants, context);"
        ))

        is_filter = "mask_t" in [out.typedef for out in outputs]
        if is_filter:
            index_of_mask_t = [out.typedef for out in outputs].index("mask_t")
            code += f"\nconst auto decision = std::get<{index_of_mask_t}>(output_container).size() ? FilterDecision::PASSED : FilterDecision::FAILED;\n"
        else:
            # always return passed if its not a filter
            code += "\nconst auto decision = FilterDecision::PASSED;\n"

        # take return values
        code += "\n".join((
            f"m_{out.typename}.put(std::move(std::get<{i}>(output_container)));"
            for i, out in enumerate(outputs)))

        code += "\nreturn decision;"

        code += "\n}\n};"
        code += f"\n\n DECLARE_COMPONENT({algorithm.name})\n"

        return code

    @staticmethod
    def generate_gaudi_wrapper(algorithm):
        # Initialize the properties
        properties_initialization = [
            f"m_algorithm.set_property_value<{algorithm.namespace}::Parameters::{p.typename}, {p.typedef}>(m_{p.typename}.value());"
            if p.scope == "algorithm" else
            f"m_algorithm.set_property_value<Allen::Algorithm::{p.typename}, {p.typedef}>(m_{p.typename}.value());"
            for p in algorithm.properties
        ]
        properties = [
            f"Gaudi::Property<{p.typedef}> m_{p.typename}{{this, {p.name}, {p.default_value}, [=](auto&) {{ {init} }}, Gaudi::Details::Property::ImmediatelyInvokeHandler{{true}}, {p.description} }};"
            for p, init in zip(algorithm.properties, properties_initialization)
        ]
        properties += [
            # a property indicating that it does not contain optionals
            'Gaudi::Property<bool> m_hasOptionals{this, "hasOptionals", false};'
        ]

        inputs = [
            parameter for parameter in algorithm.parameters
            if "input" in parameter.kind.lower()
        ]
        outputs = [
            parameter for parameter in algorithm.parameters
            if "output" in parameter.kind.lower()
        ]

        # For now, algorithms always receive as inputs RuntimeOptions and Constants,
        # therefore no Allen algorithm becomes a Producer
        if not outputs:
            is_filter = False
            base_type = "Consumer"
            include_file = "Consumer.h"
            output_type = "void"
            output_container = ""
            return_statement = ""
            operator_output_type = output_type
        else:
            include_file = "Transformer.h"
            output_types = [
                f"Allen::parameter_vector<{algorithm.namespace}::Parameters::{p.typename}::type>"
                for p in outputs
            ]

            # If there is a mask_t among the types of the outputs,
            # then classify this algorithm as a filter
            is_filter = "mask_t" in [out.typedef for out in outputs]
            if is_filter:
                base_type = "MultiTransformerFilter"
                output_type = "std::tuple<" + ",".join(output_types) + ">"
                operator_output_type = "std::tuple<bool, " + \
                    ",".join(output_types) + ">"
                output_container = "output_t output_container{};"
                index_of_mask_t = [out.typedef
                                   for out in outputs].index("mask_t")
                return_statement = f"return std::tuple_cat(std::tuple<bool>{{std::get<{index_of_mask_t}>(output_container).size()}}, output_container);"
            else:
                base_type = "MultiTransformer"
                output_type = "std::tuple<" + ",".join(output_types) + ">"
                operator_output_type = output_type
                output_container = "output_t output_container{};"
                return_statement = "return output_container;"

        base_type_namespace = "Gaudi::Functional::"
        full_base_type = base_type_namespace + base_type

        input_types = [
            f"Allen::parameter_vector<{algorithm.namespace}::Parameters::{p.typename}::type> const&"
            for p in inputs
        ]

        # RuntimeOptions and constants need to be passed as inputs to all Allen algorithms
        additional_inputs = [("runtime_options_t", "const RuntimeOptions&",
                              "runtime_options"),
                             ("constants_t", "Constants const * const &",
                              "constants")]

        inputs_tuple = ", ".join(input_types +
                                 [a[1] for a in additional_inputs])
        operator_inputs = ", ".join([
            t + " " + i.typename + "_arg" for t, i in zip(input_types, inputs)
        ] + [
            t + " " + name
            for t, name in zip([a[1] for a in additional_inputs],
                               [a[2] for a in additional_inputs])
        ])

        input_keyvals = ", ".join(
            [f'KeyValue("{p.typename}", {{""}})' for p in inputs] + [
                f"KeyValue(\"{p}\", {{\"\"}})"
                for p in [a[0] for a in additional_inputs]
            ])
        output_keyvals = ", ".join(
            [f'KeyValue("{p.typename}", {{""}})' for p in outputs])

        # Consider inputs and outputs have to be added only if they exist
        input_keyvals_list = ["{" + input_keyvals + "}"
                              ] if input_keyvals else []
        output_keyvals_list = ["{" + output_keyvals + "}"
                               ] if output_keyvals else []
        input_and_output_keyvals = ",".join(input_keyvals_list +
                                            output_keyvals_list)

        tes_wrappers_list = []
        tes_wrappers_reference_initialization_list = []
        output_container_element = 0
        for i, p in enumerate(algorithm.parameters):
            # Fetch the type of the TES wrapper for parameter p
            # Produce the initialization line for parameter p into a TESWrapper
            if p in inputs:
                tes_wrapper_type_text = "TESWrapperInput"
                parameter_variable_name = f"{p.typename}_arg"
            else:
                tes_wrapper_type_text = "TESWrapperOutput"
                parameter_variable_name = f"std::get<{output_container_element}>(output_container)"
                output_container_element += 1

            tes_wrapper_initialization = f"{{{parameter_variable_name},\"{p.typename}\"}}"
            tes_wrapper_variable_name = f"{p.typename}_wrapper"

            tes_wrappers_list.append(
                f"Allen::{tes_wrapper_type_text}<{algorithm.namespace}::Parameters::{p.typename}::type> {tes_wrapper_variable_name} {tes_wrapper_initialization};"
            )
            tes_wrappers_reference_initialization_list.append(
                f"{tes_wrapper_variable_name}")

        tes_wrappers = "\n".join(tes_wrappers_list)
        tes_wrappers_reference_initialization = ",".join(
            tes_wrappers_reference_initialization_list)
        tes_wrappers_reference = f"std::array<std::reference_wrapper<Allen::Store::BaseArgument>, {len(algorithm.parameters)}> tes_wrappers_references {{{tes_wrappers_reference_initialization}}};"

        code = "\n".join([
            f"#include \"AlgorithmConversionTools.h\"",
            f"#include <{algorithm.filename}>",
            f"#include <GaudiAlg/{include_file}>",
            "#include <GaudiAlg/FunctionalUtilities.h>",
            "#include <vector>",
            "// output type",
            f"using output_t = {output_type};",
            "// algorithm definition",
            f"using base_class_t = {full_base_type}<output_t(EventContext const&, {inputs_tuple}), Gaudi::Functional::Traits::useAlgorithm>;",
            f"struct {algorithm.name} final : base_class_t {{",
            f"StatusCode initialize() override {{",
            f"    const StatusCode sc = base_class_t::initialize();",
            f"    if ( sc.isFailure() ) return sc;",
            f"    Allen::initialize_algorithm(m_algorithm);",
            f"    m_algorithm.set_name(this->name());",
            f"    return sc;",
            f"}}",
            f"// wrapped algorithm body",
            f"{algorithm.name}( std::string const& name, ISvcLocator* pSvc )",
            f"  : {base_type}( name, pSvc, {input_and_output_keyvals} ) {{}}",
            f"// operator()",
            f"{operator_output_type} operator()(EventContext const&, {operator_inputs}) const override {{",
            output_container,
            "// TES wrappers",
            f"{tes_wrappers}",
            "// Inputs to set_arguments_size and operator()",
            f"{tes_wrappers_reference}",
            f"Allen::Context context{{}};",
            f"// set arguments size invocation",
            f"m_algorithm.set_arguments_size(tes_wrappers_references, runtime_options, *constants);",
            f"// algorithm operator() invocation",
            f"m_algorithm(tes_wrappers_references, runtime_options, *constants, context);",
            return_statement,
            f"}}",
            "private:",
            f"{algorithm.namespace}::{algorithm.name} m_algorithm{{}};",
            "\n".join(properties),
            f"}};",
            f"DECLARE_COMPONENT({algorithm.name})",
        ])

        return code

    @staticmethod
    def write_algorithms_view(algorithms, filename, default_properties):
        from subprocess import (PIPE, run)

        # Run the default_properties executable to get a JSON
        # representation of the default values of all properties of
        # all algorithms
        p = run(
            [default_properties],
            stdout=PIPE,
            input=';'.join([
                "{}::{}".format(a.namespace, a.name) for a in parsed_algorithms
            ]),
            encoding='ascii')

        default_properties = None
        if p.returncode == 0:
            default_properties = json.loads(p.stdout)
        else:
            print("Failed to obtain default property values")
            sys.exit(-1)

        s = AllenCore.write_preamble()
        for algorithm in parsed_algorithms:
            tn = "{}::{}".format(algorithm.namespace, algorithm.name)
            s += AllenCore.write_algorithm_code(algorithm,
                                                default_properties[tn])
        with open(filename, "w") as f:
            f.write(s)

    @staticmethod
    def write_gaudi_algorithms(algorithms,
                               algorithm_wrappers_folder,
                               write_files=True):
        algorithms_generated_filenames = []
        for alg in algorithms:
            if not [
                    var
                    for var in alg.parameters if var.aggregate or var.optional
            ]:
                code = AllenCore.generate_gaudi_wrapper(alg)
            else:
                code = AllenCore.generate_gaudi_wrapper_for_aggregate(alg)
            output_filename = algorithm_wrappers_folder + "/" + alg.name + "_gaudi.cpp"
            if write_files:
                with open(output_filename, "w") as f:
                    f.write(code)
            algorithms_generated_filenames.append(output_filename)
        return algorithms_generated_filenames

    @staticmethod
    def write_algorithm_filename_list(filenames,
                                      output_filename,
                                      separator=";"):
        s = separator.join([a for a in filenames])
        with open(output_filename, "w") as f:
            f.write(s)

    @staticmethod
    def write_algorithms_db(algorithms, filename):
        code = "\n".join(("#include <AlgorithmDB.h>", "\n"))
        for alg in algorithms:
            code += f"namespace {alg.namespace} {{ struct {alg.name}; }}\n"
        code += "\nAllen::TypeErasedAlgorithm instantiate_allen_algorithm(const ConfiguredAlgorithm& alg) {\n"
        for i, alg in enumerate(algorithms):
            if i == 0:
                code += f"  if (alg.id == \"{alg.namespace}::{alg.name}\") {{\n"
            else:
                code += f"  }} else if (alg.id == \"{alg.namespace}::{alg.name}\") {{\n"
            code += f"    return Allen::instantiate_algorithm<{alg.namespace}::{alg.name}>(alg.name);\n"
        code += "\n".join(
            ("  } else {", "    throw AlgorithmNotExportedException{alg.id};",
             "  }", "}"))
        with open(filename, "w") as f:
            f.write(code)

    @staticmethod
    def write_extern_lines(algorithms, filename, separable_compilation):
        selection_algorithms = [
            a for a in algorithms if a.scope == "SelectionAlgorithm"
        ]
        code = "\n".join(("#pragma once", "", "#include \"BackendCommon.h\"",
                          "\n"))
        for alg in selection_algorithms:
            code += "\n".join(
                (f"namespace {alg.namespace} {{", f"  struct {alg.name};",
                 "  struct Parameters;", "}\n"))
        code += "\n"
        if separable_compilation:
            for alg in selection_algorithms:
                code += f"extern template __device__ void process_line<{alg.namespace}::{alg.name}, {alg.namespace}::Parameters>(char*, bool*, unsigned*, Allen::IMultiEventContainer**, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, const unsigned);\n"
            code += "\n"
            for alg in selection_algorithms:
                code += f"extern template void line_output_monitor<{alg.namespace}::{alg.name}, {alg.namespace}::Parameters>(char*, const RuntimeOptions&, const Allen::Context&);\n"
        code += "\nconstexpr auto line_strings = {\n"
        for i, alg in enumerate(selection_algorithms):
            code += f"  \"{alg.name}\""
            if i != len(selection_algorithms) - 1:
                code += ",\n"
        code += "\n};\n\n"
        code += "__device__ inline void invoke_line_functions(unsigned index, char* a, bool* b, unsigned* c, Allen::IMultiEventContainer** d, unsigned e, unsigned f, unsigned g, unsigned h, unsigned i, unsigned j, const unsigned k) {\n"
        code += f"  assert(index < {len(selection_algorithms)});\n"
        code += "  switch (index) {\n"
        for i, alg in enumerate(selection_algorithms):
            code += f"    case {i}: process_line<{alg.namespace}::{alg.name}, {alg.namespace}::Parameters>(a, b, c, d, e, f, g, h, i, j, k); break;\n"
        code += "  }\n}\n\n"
        code += f"constexpr std::array<void(*)(char*, const RuntimeOptions&, const Allen::Context&), {len(selection_algorithms)}> line_output_monitor_functions = {{\n"
        for i, alg in enumerate(selection_algorithms):
            code += f"  line_output_monitor<{alg.namespace}::{alg.name}, {alg.namespace}::Parameters>"
            if i != len(selection_algorithms) - 1:
                code += ",\n"
        code += "\n};\n"
        # void inline invoke_output_monitor(const char* arg_ref, const RuntimeOptions& runtime_options, const Allen::Context& context) {
        with open(filename, "w") as f:
            f.write(code)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=
        'Parse the Allen codebase and generate a python representation of all algorithms.'
    )

    parser.add_argument(
        '--filename',
        nargs='?',
        type=str,
        default="algorithms.py",
        help='output filename')
    parser.add_argument(
        '--prefix_project_folder',
        nargs='?',
        type=str,
        default="..",
        help='project location')
    parser.add_argument(
        "--algorithm_wrappers_folder",
        nargs="?",
        type=str,
        default="",
        help="converted algorithms folder")
    parser.add_argument(
        "--parsed_algorithms",
        nargs="?",
        type=str,
        default="",
        help="location of parsed algorithms")
    parser.add_argument(
        "--default_properties",
        nargs="?",
        type=str,
        default="",
        help="location of default_properties executable")
    parser.add_argument(
        "--generate",
        nargs="?",
        type=str,
        default="views",
        choices=[
            "parsed_algorithms", "views", "wrapperlist", "wrappers", "db",
            "extern_lines", "extern_lines_nosepcomp", "algorithm_headers_list"
        ],
        help="action that will be performed")

    args = parser.parse_args()
    prefix_folder = args.prefix_project_folder + "/"
    if args.generate == "parsed_algorithms":
        algorithm_files = Parser().find_algorithm_files(prefix_folder)
        parsed_algorithms = Parser().parse_all(algorithm_files, prefix_folder)
        with open(args.filename, "wb") as f:
            pickle.dump(parsed_algorithms, f)
    elif args.generate == "algorithm_headers_list":
        # Write list of files including algorithm definitions
        algorithm_headers_list = Parser().find_algorithm_files(prefix_folder)
        AllenCore.write_algorithm_filename_list(algorithm_headers_list,
                                                args.filename)
    else:

        if args.parsed_algorithms:
            # Load pregenerated parsed_algorithms
            with open(args.parsed_algorithms, "rb") as f:
                parsed_algorithms = pickle.load(f)
        else:
            # Otherwise generate parsed_algorithms on the fly
            algorithm_files = Parser().find_algorithm_files(prefix_folder)
            parsed_algorithms = Parser().parse_all(algorithm_files,
                                                   prefix_folder)

        if args.generate == "views":
            # Generate algorithm python views
            AllenCore.write_algorithms_view(parsed_algorithms, args.filename,
                                            args.default_properties)
        elif args.generate == "wrapperlist":
            # Generate Gaudi wrapper filenames
            gaudi_wrapper_filenames = AllenCore.write_gaudi_algorithms(
                parsed_algorithms,
                args.algorithm_wrappers_folder,
                write_files=False)
            # Write algorithm list in txt format for CMake
            AllenCore.write_algorithm_filename_list(gaudi_wrapper_filenames,
                                                    args.filename)
        elif args.generate == "wrappers":
            # Write Gaudi wrappers on top of all algorithms
            AllenCore.write_gaudi_algorithms(parsed_algorithms,
                                             args.algorithm_wrappers_folder)
        elif args.generate == "db":
            # Generate Allen algorithms DB
            AllenCore.write_algorithms_db(parsed_algorithms, args.filename)
        elif args.generate == "extern_lines":
            # Write extern lines header file
            AllenCore.write_extern_lines(parsed_algorithms, args.filename,
                                         True)
        elif args.generate == "extern_lines_nosepcomp":
            # Write extern lines header file, without separable compilation
            AllenCore.write_extern_lines(parsed_algorithms, args.filename,
                                         False)
