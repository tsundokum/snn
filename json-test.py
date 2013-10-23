#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

import json

with open('resources/RogersMcClelland08_ru.json') as f:
    txt = f.read()
    print(txt)
    j = json.loads(txt)
    print(j['relations'][0])