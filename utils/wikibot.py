#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jyao
"""
import pywikibot

class Wikibot:
    """
    A wrapper for pywikibot with caching
    """
    def __init__(self):
        self.site = pywikibot.Site('en', 'wikipedia')
        self.cache = {}

    def search_in_page(self, text, title):
        """
        text: str or list of str
        Return:
            return True if page contains at least one element in text
        """
        if type(text) != list:
            text = [text]
        if title not in self.cache:
            page = pywikibot.Page(self.site, title)
            for s in text:
                if s in page.text:
                    self.cache[title] = True
                    return self.cache[title]
            # if none of text found in page
            self.cache[title] = False
        return self.cache[title]
    