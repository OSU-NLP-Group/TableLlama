""" Utility functions for datadumping."""
import unicodedata
import re
from openpyxl.utils import get_column_letter, column_index_from_string
import functools


# Compare and sort cells
def find_column(coord):
    """ Parse column letter from 'E3'. """
    return re.findall('[a-zA-Z]+', coord)


def find_row(coord):
    """ Parse row number from 'E3'. """
    return re.findall('[0-9]+', coord)


def cell_compare(cell1, cell2):
    """ Compare cell coord by row, then by column."""
    col1, col2 = find_column(cell1)[0], find_column(cell2)[0]
    row1, row2 = find_row(cell1)[0], find_row(cell2)[0]
    if int(row1) < int(row2):
        return -1
    elif int(row1) > int(row2):
        return 1
    else:
        if column_index_from_string(col1) < column_index_from_string(col2):
            return -1
        else:
            return 1


def linked_cell_compare(linked_cell_a, linked_cell_b):
    """ Compare answer cell coord by row, then by column."""
    if isinstance(linked_cell_a[0], str) and isinstance(linked_cell_b[0], str):
        coord_a, coord_b = eval(linked_cell_a[0]), eval(linked_cell_b[0])
    else:
        coord_a, coord_b = linked_cell_a[0], linked_cell_b[0]
    if coord_a[0] < coord_b[0]:
        return -1
    elif coord_a[0] > coord_b[0]:
        return 1
    else:
        if coord_a[1] < coord_b[1]:
            return -1
        else:
            return 1


def sort_region_by_coord(cells):
    """ Sort cells by coords, according to cell_compare(). """
    cell_list = sorted(cells, key=functools.cmp_to_key(cell_compare))
    cell_matrix = []
    last_row = None
    for cell in cell_list:
        col, row = find_column(cell), find_row(cell)
        if row == last_row:
            cell_matrix[-1].append(cell)
        else:
            last_row = row
            cell_matrix.append([cell])
    return cell_list, cell_matrix


# --------------------------------------------
# Normalize and Inferring Types.
def normalize(x):
    """ Normalize header string. """
    # Copied from WikiTableQuestions dataset official evaluator.
    if x is None:
        return None
    # Remove diacritics
    x = ''.join(c for c in unicodedata.normalize('NFKD', x)
                if unicodedata.category(c) != 'Mn')
    # Normalize quotes and dashes
    x = re.sub("[‘’´`]", "'", x)
    x = re.sub("[“”]", "\"", x)
    x = re.sub("[‐‑‒–—−]", "-", x)
    while True:
        old_x = x
        # Remove citations
        x = re.sub("((?<!^)\[[^\]]*\]|\[\d+\]|[•♦†‡*#+])*$", "", x.strip())
        # Remove details in parenthesis
        x = re.sub("(?<!^)( \([^)]*\))*$", "", x.strip())
        # Remove outermost quotation mark
        x = re.sub('^"([^"]*)"$', r'\1', x.strip())
        if x == old_x:
            break
    # Remove final '.'
    if x and x[-1] == '.':
        x = x[:-1]
    # Collapse whitespaces and convert to lower case
    x = re.sub('\s+', ' ', x, flags=re.U).lower().strip()
    return x


def naive_str_to_float(string):
    """ A naive way to convert str to float, if convertable."""
    sanitized = string
    try:
        if sanitized[0] == '(':
            sanitized = sanitized[1:]
        if (sanitized[-1] == '%') or (sanitized[-1] == ')'):
            sanitized = sanitized[: -1]
        sanitized = sanitized.replace(',', '')
        new = float(sanitized)
        return new
    except:
        return normalize(string)
