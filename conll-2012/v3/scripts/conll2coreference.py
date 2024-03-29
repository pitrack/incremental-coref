#!/usr/bin/env python

#---- standard library imports ----#
import sys

# version check
if(not (sys.hexversion >= 0x2050000)):
    sys.exit("\n\nplease use python version >= 2.5\n\n")

import string
import re
import math
import os
import time
import getopt
import zlib
import gzip
import codecs
import optparse
import collections
import ConfigParser
from optparse import OptionParser


PART_COLUMN=1
WORD_COLUMN=3










def error(error_string, terminate_program=True, current_frame=False):
    """Print error messages to stderr, optionally sys.exit"""

    if(current_frame == False):
        pretty_error_string = """

--------------------------------------------------------------------------------
                                       ERROR
--------------------------------------------------------------------------------
%s
--------------------------------------------------------------------------------

""" % (error_string)
    else:
        pretty_error_string = """

--------------------------------------------------------------------------------
                                       ERROR
--------------------------------------------------------------------------------
FILE: %s
LINE: %s
--------------------------------------------------------------------------------
%s
--------------------------------------------------------------------------------

""" % (current_frame.f_code.co_filename, current_frame.f_lineno, error_string)

    sys.stderr.write(pretty_error_string)

    if(terminate_program == True):
        sys.exit(1)










def parse_cfg_args(arg_list):
    """Parse command-line style config settings to a dictionary.

    If you want to override configuration file values on the command
    line or set ones that were not set, this should make it simpler.
    Given a list in format [section.key=value, ...] return a
    dictionary in form { (section, key): value, ...}.

    So we might have:

    .. code-block:: python

      ['corpus.load=english-mz',
       'corpus.data_in=/home/user/corpora/ontonotes/data/']

    we would then return the dictionary:

    .. code-block:: python

      { ('corpus', 'load') : 'english-mz',
        ('corpus', 'data_in') : '/home/user/corpora/ontonotes/data/' }

    See also :func:`load_config` and :func:`load_options`

    """

    if not arg_list:
        return {}

    config_append = {}

    for arg in arg_list:
        if len(arg.split("=")) != 2 or len(arg.split("=")[0].split('.')) != 2:
            raise Exception("Invalid argument; not in form section.key=value : " + arg)

        key, value = arg.split("=")
        config_append[tuple(key.split("."))] = value

    return config_append














__registered_config_options = collections.defaultdict( dict )

def required_config_sections():
    return [section for section in __registered_config_options if
            [True for value in __registered_config_options[section]
             if __registered_config_options[section][value][3]]] # section_required












def load_options(parser=None, argv=[], positional_args=True):
    """ parses sys.argv, possibly exiting if there are mistakes

    If you set parser to a ConfigParser object, then you have control
    over the usage string and you can prepopulate it with options you
    intend to use.  But don't set a ``--config`` / ``-c`` option;
    load_options uses that to find a configuration file to load

    If a parser was passed in, we return ``(config, parser, [args])``.
    Otherwise we return ``(config, [args])``.  Args is only included
    if ``positional_args`` is True and there are positional arguments

    See :func:`load_config` for details on the ``--config`` option.

    """

    def is_config_appender(arg):
        return "." in arg and "=" in arg and arg.find(".") < arg.find("=")

    parser_passed_in=parser
    if not parser:
        parser = OptionParser()

    parser.add_option("-c", "--config", help="the path to a config file to read options from")

    if argv:
        options, args = parser.parse_args(argv)
    else:
        options, args = parser.parse_args()

    config = load_config(options.config, [a for a in args if is_config_appender(a)])

    other_args = [a for a in args if not is_config_appender(a)]

    return_list = [config]
    if parser_passed_in:
        return_list.append(options)
    if other_args:
        if positional_args:
            return_list.append(other_args)
        else:
            raise Exception("Arguments %s not understood" % other_args)
    else:
        if positional_args:
            raise Exception("This program expects one or more positional arguments that are missing")

    if len(return_list) == 1:
        return return_list[0]
    else:
        return tuple(return_list)














class FancyConfigParserError(Exception):
    """ raised by :class:`FancyConfigParser` when used improperly """

    def __init__(self, vals):
        Exception.__init__(self, 'Config usage must be in the form "config[\'section\', \'item\']". '
                           'Given something more like "config[%s]".' % (", ".join("%r"%v for v in vals)))










class FancyConfigParser(ConfigParser.SafeConfigParser):
    """ make a config parser with support for config[section, value]

    raises :class:`FancyConfigParserError` on improper usage.

    """

    def __getitem__(self, vals):
        try:
            section, item = vals
        except (ValueError, TypeError):
            raise FancyConfigParserError(vals)
        return self.get(section, item)


    def __setitem__(self, vals, value):
        try:
            section, item = vals
        except (ValueError, TypeError):
            raise FancyConfigParserError(vals)
        return self.set(section, item, value)

    def __delitem__(self, vals):
        try:
            section, item = vals
        except (ValueError, TypeError):
            raise FancyConfigParserError(vals)

        self.remove_option(section, item)










def load_config(cfg_name=None, config_append=[]):
    """ Load a configuration file to memory.

    The given configuration file name can be a full path, in which
    case we simply read that configuration file.  Otherwise, if you
    give 'myconfig' or something similar, we look in the current
    directory and the home directory.  We also look to see if files
    with this name and extension '.conf' exist.  So for 'myconfig' we
    would look in the following places:

     * ./myconfig
     * ./myconfig.conf
     * [home]/.myconfig
     * [home]/.myconfig.conf

    Once we find the configuration, we load it.  We also extend
    ConfigParser to support ``[]`` notation.  So you could look up key
    ``k`` in section ``s`` with ``config[s,k]``.  See
    :func:`FancyConfigParser` .

    If config_append is set we use :func:`parse_cfg_args` and add any
    values it creates to the config object.  These values override any
    previous ones.

    """

    config = FancyConfigParser()

    if cfg_name:
        config_locs = [cfg_name + '.conf',
                       os.path.expanduser('~/.' + cfg_name + '.conf'),
                       cfg_name,
                       os.path.expanduser('~/.' + cfg_name)]
        l = config.read(config_locs)
        if not l:
            raise Exception("Couldn't find config file.  Looked in:" +
                            "".join(["\n - " + c for c in config_locs]) +
                            "\nto no avail.")


    for (section, key_name), value in parse_cfg_args(config_append).iteritems():
        if not config.has_section(section):
            config.add_section(section)
        config.set(section, key_name, value)

    problems = []
    for section in config.sections():
        if not is_config_section_registered(section):
            on.common.log.status("Ignoring unknown configuration section", section)
            continue
        for option in config.options(section):
            if not is_config_registered(section, option):
                problems.append("Unknown configuration variable %s.%s" % (section, option))
                continue

            value = config.get(section, option)
            allowed = allowed_config_values(section, option)
            multiple = allow_multiple_config_values(section, option)

            values = value.split() if multiple else [value]
            for value in values:
                if allowed and not value in allowed:
                    problems.append("Illegal value '%s' for configuration variable %s.%s.  Permitted values are: %s" %
                                    (value, section, option, ", ".join(["'%s'" % x for x in allowed])))

        for option in required_config_options(section):
            if not config.has_option(section, option):
                problems.append("Required configuration variable %s.%s is absent" % (section, option))

    for section in required_config_sections():
        if not config.has_section(section):
            problems.append("Required configuration section %s is absent" % section)

    if problems:
        print_config_docs()

        on.common.log.status("Configuration Problems:")
        for problem in problems:
            on.common.log.status("  " + problem)

        sys.exit(-1)

    return config












def make_sgml_safe(s, reverse=False, keep_turn=True):
    """ return a version of the string that can be put in an sgml document

    This means changing angle brackets and ampersands to '-LAB-',
    '-RAB-', and '-AMP-'.  Needed for creating ``.name`` and
    ``.coref`` files.

    If keep_turn is set, <TURN> in the input is turned into [TURN], not turned into -LAB-TURN-RAB-

    """

    if not reverse and keep_turn:
        s = s.replace("<TURN>", "[TURN]")

    for f, r in [("<", "-LAB-"),
                 (">", "-RAB-"),
                 ("&", "-AMP-")]:
        if reverse:
            r, f = f, r
        s = s.replace(f, r)

    return s







class link:

    def __init__(self, id, start, end):
        self.id = id
        self.start = start
        self.end = end


    def __repr__(self):
        return "<link: %s:%s:%s>" % (self.id, self.start, self.end)










def link_compare(a_link, b_link):
    if(a_link.start != b_link.start):
        return cmp(a_link.start, b_link.start)
    else:
        return cmp(a_link.end, b_link.end)










class coreference_tagged_sentence:

    def __init__(self, r_c_matrix):
        self.part_number = r_c_matrix[0][0]
        self.words = []
        self.links = []               # list of all links in this sentence
        self.chain_hash = {}          # hash of all links in this sentence



        chain_start_hash = {}


        #rows = []
        #for tuple in r_c_matrix:
        #    rows.append(" ".join(tuple))
        #
        #on.common.util.pretty_print_table(rows)
        #print


        for r_i, r in enumerate(r_c_matrix):

            assert self.part_number == r[0], "all rows should contain the same part number"
            self.words.append(r[1])

            # process encoded chain
            encoded_chain=r[2]
            bits = encoded_chain.split("|")


            ids = []
            for i in range(0, len(bits)):
                id = bits[i].replace("(", "")
                id = id.replace(")", "")
                ids.append(id)

            assert len(ids) == len(bits), "the length of ids and bits should be the same"



            for i in range(0, len(bits)):
                if(bits[i].startswith("(")):
                    if(not chain_start_hash.has_key(ids[i])):
                        chain_start_hash[ids[i]] = []
                    chain_start_hash[ids[i]].append(r_i)

                if(bits[i].endswith(")")):

                    if(not chain_start_hash.has_key(ids[i])):
                        print chain_start_hash
                        raise Exception("problem, found link end without a start")



                    try:
                        a_link = link(ids[i], chain_start_hash[ids[i]].pop(), r_i)



                        self.links.append(a_link)

                        if(not self.chain_hash.has_key(ids[i])):
                            self.chain_hash[ids[i]] = []

                        self.chain_hash[ids[i]].append(a_link)
                    except:
                        sys.stderr.write("WARNING: dropping link with id [%s]" % (ids[i]))





        for k, v in chain_start_hash.iteritems():
            if( len(v) != 0):
                raise Exception("all the lists in the start hash should be empty")


        self.links.sort(link_compare)










    def __repr__(self):

        coref_tagged_words = []
        coref_tagged_words.extend(self.words)

        # make words sgml safe
        for i in range(0, len(coref_tagged_words)):
            coref_tagged_words[i] = make_sgml_safe(coref_tagged_words[i])


        for a_link in self.links:
            coref_tagged_words[a_link.start] = """<COREF ID="%s" TYPE="IDENT">%s""" % (a_link.id, coref_tagged_words[a_link.start])
            coref_tagged_words[a_link.end] = "%s</COREF>" % (coref_tagged_words[a_link.end])


        return "%s" % (" ".join(coref_tagged_words))





def expand_document_id(document_id, language):

    if language == "english":
        abbr_language = "en"
    elif language == "chinese":
        abbr_language = "ch"
    elif language == "arabic":
        abbr_language = "ar"


    file_bit=document_id[-4:]
    genre_bit, source_bit, ignore = document_id.split("/", 2)
    constant="%s@on" % (abbr_language)
    return "%s@%s@%s@%s@%s" % (document_id, file_bit, source_bit, genre_bit, constant)









def main():
    # total number of expected actual arguments, not counting the command itself
    required_number_of_args = 1

    o_parser = optparse.OptionParser()
    o_parser.set_defaults(DEBUG=False)
    o_parser.set_defaults(LANGUAGE=None)

    o_parser.add_option("-d", "--debug", action="store_true", dest="DEBUG", help="Set debug mode on")
    o_parser.add_option("-l", "--language", action="store", dest="LANGUAGE", help="Set language")
    o_parser.add_option("-o", "--option", help="perform the specified task.  this can be 'pre-process' or 'decode'")

    if(required_number_of_args > 0):
        c_config, o_options, o_args = load_options(parser=o_parser)

        if(len(o_args) != required_number_of_args):
            error("please specify %s arguments" % (required_number_of_args))
    else:
        c_config, o_options = load_options(parser=o_parser, positional_args=False)



    legal_options = []
    if(legal_options != []
       and
       o_options.option not in legal_options):
        error("please specify one of %s options" % (" ".join(legal_options)))



    if o_options.LANGUAGE is None:
        error("please specify language using -l option")


    r_c_matrix = []
    sentence_index = 0
    if(required_number_of_args > 0):
        file = open(o_args[0])
        file_line = file.readline()
        while( file_line != "" ):
            #---- start processing here ----#
            file_line = file_line.strip()

            if(file_line.startswith("#begin")):
                bits = file_line.split()
                document_id=bits[2].replace("(", "").replace(");","")
                part_number=bits[-1]

                if(part_number != "000"):
                    print '</TEXT>\n<TEXT PARTNO="%s">' % (part_number)
                else:
                    print '<DOC DOCNO="%s">\n<TEXT PARTNO="%s">' % (expand_document_id(document_id, o_options.LANGUAGE), part_number)

            elif(file_line.startswith("#end")):
                pass

            elif(file_line == ""):
                sentence_index = sentence_index + 1
                #print "sentence:", sentence_index
                a_coreference_tagged_sentence = coreference_tagged_sentence(r_c_matrix)
                print a_coreference_tagged_sentence
                r_c_matrix = []

            else:
                columns = file_line.split()
                (part_number, word, encoded_chain) = columns[PART_COLUMN], columns[WORD_COLUMN], columns[-1]
                if o_options.LANGUAGE == "arabic":
                    r_c_matrix.append([part_number, re.sub("#.*", "", word), encoded_chain])
                else:
                    r_c_matrix.append([part_number, word, encoded_chain])


            file_line = file.readline()

        print "</TEXT>\n</DOC>"

        #---- close the file ----#
        if(file != sys.stdin):
            file.close()

if __name__ == '__main__':
    main()

