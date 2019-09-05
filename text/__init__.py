""" from https://github.com/keithito/tacotron """
import re
from text import cleaners
from text.symbols import eng_symbols, kor_symbols
from text.unicode import join_jamos

from jamo import j2hcj
from sys import stderr

import os
import json

cleaner_names = ["korean_cleaners"]
_ROOT = os.path.abspath(os.path.dirname(__file__))

# Mappings from symbol to numeric ID and vice versa:
symbols = ""
_symbol_to_id = {}
_id_to_symbol = {}

# Regular expression matching text enclosed in curly braces:
_curly_re = re.compile(r'(.*?)\{(.+?)\}(.*)')

with open(os.path.join(_ROOT, "decompositions.json"), 'r') as namedata:
  _JAMO_TO_COMPONENTS = json.load(namedata)
_COMPONENTS_REVERSE_LOOKUP = {tuple(comps): char for char, comps in _JAMO_TO_COMPONENTS.items()}


class InvalidJamoError(Exception):
  """jamo is a U+11xx codepoint."""

  def __init__(self, message, jamo):
    super(InvalidJamoError, self).__init__(message)
    self.jamo = hex(ord(jamo))
    print("Could not parse jamo: U+{code}".format(code=self.jamo[2:]),
          file=stderr)

def compose_jamo(*parts):
  """Return the compound jamo for the given jamo input.
  Integers corresponding to U+11xx jamo codepoints, U+11xx jamo
  characters, or HCJ are valid inputs.

  Outputs a one-character jamo string.
  """
  # Internally, we convert everything to a jamo char,
  # then pass it to _jamo_to_hangul_char
  # NOTE: Relies on hcj_to_jamo not strictly requiring "position" arg.
  for p in parts:
    if not (type(p) == str and len(p) == 1 and 2 <= len(parts) <= 3):
      raise TypeError("compose_jamo() expected 2-3 single characters " +
                      "but received " + str(parts),
                      '\x00')
  hcparts = [j2hcj(_) for _ in parts]
  hcparts = tuple(hcparts)
  if hcparts in _COMPONENTS_REVERSE_LOOKUP:
    return _COMPONENTS_REVERSE_LOOKUP[hcparts]
  raise InvalidJamoError(
    "Could not synthesize characters to compound: " + ", ".join(
      str(_) + "(U+" + str(hex(ord(_)))[2:] +
      ")" for _ in parts), '\x00')


def change_symbol(cleaner_names):
  symbols = ""
  global _symbol_to_id
  global _id_to_symbol
  if cleaner_names == ["english_cleaners"]: symbols = eng_symbols
  if cleaner_names == ["korean_cleaners"]: symbols = kor_symbols

  _symbol_to_id = {s: i for i, s in enumerate(symbols)}
  _id_to_symbol = {i: s for i, s in enumerate(symbols)}

change_symbol(cleaner_names)

def text_to_sequence(text, cleaner_names):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.

    The text can optionally have ARPAbet sequences enclosed in curly braces embedded
    in it. For example, "Turn left on {HH AW1 S S T AH0 N} Street."

    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through

    Returns:
      List of integers corresponding to the symbols in the text
  '''
  sequence = []
  change_symbol(cleaner_names)
  # Check for curly braces and treat their contents as ARPAbet:
  while len(text):
    m = _curly_re.match(text)
    try:
      if not m:
        sequence += _symbols_to_sequence(_clean_text(text, cleaner_names))
        break
      sequence += _symbols_to_sequence(_clean_text(m.group(1), cleaner_names))
      sequence += _arpabet_to_sequence(m.group(2))
      text = m.group(3)
    except:
      print(text)
      exit()
  # Append EOS token
  #if cleaner_names == ["korean_cleaners"]: sequence.append(_symbol_to_id['~'])
  return sequence


def sequence_to_text(sequence):
  '''Converts a sequence of IDs back to a string'''
  result = ''
  for symbol_id in sequence:
    if symbol_id in _id_to_symbol:
      s = _id_to_symbol[symbol_id]
      # Enclose ARPAbet back in curly braces:
      if len(s) > 1 and s[0] == '@':
        s = '{%s}' % s[1:]
      result += s
  return result.replace('}{', ' ')


def _clean_text(text, cleaner_names):
  for name in cleaner_names:
    cleaner = getattr(cleaners, name)
    if not cleaner:
      raise Exception('Unknown cleaner: %s' % name)
    text = cleaner(text)
  return text


def _symbols_to_sequence(symbols):
  return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s)]


def _arpabet_to_sequence(text):
  return _symbols_to_sequence(['@' + s for s in text.split()])

def _should_keep_symbol(s):
  return s in _symbol_to_id and s is not '_' and s is not '~'

if __name__ == "__main__":
  print(text_to_sequence('this is test sentence.? ', ['english_cleaners']))
  print(text_to_sequence('테스트 문장입니다.? ', ['korean_cleaners']))
  print(_clean_text('AB테스트 문장입니다.? ', ['korean_cleaners']))
  print(_clean_text('mp3 파일을 홈페이지에서 다운로드 받으시기 바랍니다.',['korean_cleaners']))
  print(_clean_text("마가렛 대처의 별명은 '철의 여인'이었다.", ['korean_cleaners']))
  print(_clean_text("제 전화번호는 01012345678이에요.", ['korean_cleaners']))
  print(_clean_text("‘아줌마’는 결혼한 여자를 뜻한다.", ['korean_cleaners']))
  print(text_to_sequence("‘아줌마’는 결혼한 여자를 뜻한다.", ['korean_cleaners']))

