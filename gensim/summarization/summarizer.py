#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

import logging
from gensim.parsing.preprocessing import preprocess_documents
from gensim.summarization.pagerank_weighted import pagerank_weighted as _pagerank
from gensim.summarization.commons import build_graph as _build_graph
from gensim.summarization.commons import remove_unreachable_nodes as _remove_unreachable_nodes
from gensim.summarization.bm25 import get_bm25_weights as _bm25_weights
from gensim.summarization.syntactic_unit import SyntacticUnit
from gensim.corpora import Dictionary
from math import log10 as _log10
from six.moves import xrange


INPUT_MIN_LENGTH = 10

WEIGHT_THRESHOLD = 1.e-3

logger = logging.getLogger(__name__)


def _set_graph_edge_weights(graph):
    documents = graph.nodes()
    weights = _bm25_weights(documents)

    for i in xrange(len(documents)):
        for j in xrange(len(documents)):
            if i == j or weights[i][j] < WEIGHT_THRESHOLD:
                continue

            sentence_1 = documents[i]
            sentence_2 = documents[j]

            edge_1 = (sentence_1, sentence_2)
            edge_2 = (sentence_2, sentence_1)

            if not graph.has_edge(edge_1):
                graph.add_edge(edge_1, weights[i][j])
            if not graph.has_edge(edge_2):
                graph.add_edge(edge_2, weights[j][i])

    # Handles the case in which all similarities are zero.
    # The resultant summary will consist of random sentences.
    if all(graph.edge_weight(edge) == 0 for edge in graph.edges()):
        _create_valid_graph(graph)


def _create_valid_graph(graph):
    nodes = graph.nodes()

    for i in xrange(len(nodes)):
        for j in xrange(len(nodes)):
            if i == j:
                continue

            edge = (nodes[i], nodes[j])

            if graph.has_edge(edge):
                graph.del_edge(edge)

            graph.add_edge(edge, 1)


def _build_dictionary(sentences):
    tokens = []
    for s in sentences:
        if s['tokens'] != None:
            tokens.append(s['tokens'])
    return Dictionary(tokens)


def summarize_corpus(hashable_corpus, ratio=0.2):
    """
    Returns a list of the most important documents of a corpus using a
    variation of the TextRank algorithm.
    The input must have at least INPUT_MIN_LENGTH (%d) documents for the
    summary to make sense.

    The length of the output can be specified using the ratio parameter,
    which determines how many documents will be chosen for the summary
    (defaults at 20%% of the number of documents of the corpus).

    The most important documents are returned as a list sorted by the
    document score, highest first.

    """

    # If the corpus is empty, the function ends.
    if len(hashable_corpus) == 0:
        logger.warning("Input corpus is empty.")
        return

    # Warns the user if there are too few documents.
    if len(hashable_corpus) < INPUT_MIN_LENGTH:
        logger.warning("Input corpus is expected to have at least " + str(INPUT_MIN_LENGTH) + " documents.")

    graph = _build_graph(hashable_corpus)
    _set_graph_edge_weights(graph)
    _remove_unreachable_nodes(graph)

    # Cannot calculate eigenvectors if number of unique words in text < 3. Warns user to add more text. The function ends.
    if len(graph.nodes()) < 3:
        logger.warning("Please add more sentences to the text. The number of reachable nodes is below 3")
        return

    return _pagerank(graph)


def summarize(in_sents, ratio=0.2, word_count=None, split=False):
    """
    Returns a summarized version of the given text using a variation of
    the TextRank algorithm.
    The input must be longer than INPUT_MIN_LENGTH sentences for the
    summary to make sense and must be given as a string.

    The output summary will consist of the most representative sentences
    and will also be returned as a string, divided by newlines. If the
    split parameter is set to True, a list of sentences will be
    returned.

    The length of the output can be specified using the ratio and
    word_count parameters:
        ratio should be a number between 0 and 1 that determines the
    percentage of the number of sentences of the original text to be
    chosen for the summary (defaults at 0.2).
        word_count determines how many words will the output contain.
    If both parameters are provided, the ratio will be ignored.
    """

    num_ranked = 0
    sentences = []
    processed_sents = [' '.join(s) for s in preprocess_documents(in_sents)]
    for i in range(len(in_sents)):

        tokens = None
        if processed_sents[i] != '':
            tokens = SyntacticUnit(in_sents[i], processed_sents[i]).token.split()
            num_ranked = num_ranked + 1

        sentences.append({
            'corpus': None,
            'sequence': i + 1,
            'rank': 0,
            'text': in_sents[i],
            'tokens': tokens
        })

    # If no sentence could be identified, the function ends.
    if num_ranked == 0:
        logger.warning("Input text is empty.")
        return

    # If only one sentence is present, the function raises an error (Avoids ZeroDivisionError).
    if num_ranked == 1:
        raise ValueError("input must have more than one sentence")

    # Warns if the text is too short.
    if num_ranked < INPUT_MIN_LENGTH:
        logger.warning("Input text is expected to have at least " + str(INPUT_MIN_LENGTH) + " sentences.")

    hashable_corpus = []
    dictionary = _build_dictionary(sentences)
    for s in sentences:
        if s['tokens'] != None:
            s['corpus'] = tuple(dictionary.doc2bow(s['tokens']))
            hashable_corpus.append(s['corpus'])

    pagerank_scores = summarize_corpus(hashable_corpus, ratio=ratio if word_count is None else 1)

    for s in sentences:
        if s['corpus'] in pagerank_scores:
            s['rank'] = pagerank_scores[s['corpus']][0]

    return sentences
