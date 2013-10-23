#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import json


def read_all_propositions(file_name):
    with open(file_name) as f:
        root_json = json.load(f)
        propositions = []
        for item in root_json["items"]:
            for relation_info in root_json["relations"]:
                relation = relation_info["relation"]
                attrs = relation_info["attributes"]
                for attr in attrs:
                    propositions.append((item, relation, attr))
    return propositions


if __name__ == "__main__":
    propositions = read_all_propositions("resources/RogersMcClelland08.json")
    random.shuffle(propositions)
    print "Read %d propositions" % len(propositions)
