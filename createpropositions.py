#!/usr/bin/env python
# -*- coding: utf-8 -*-

import yaml
import random


def read_all_propositions(file_name):
    with open(file_name) as f:
        root_yaml = yaml.load(f)
        propositions = []
        for item in root_yaml["items"]:
            for relation_info in root_yaml["relations"]:
                relation = relation_info["relation"]
                attrs = relation_info["attributes"]
                for attr in attrs:
                    propositions.append((item, relation, attr))
    return propositions


if __name__ == "__main__":
    propositions = read_all_propositions("resources/RogersMcClelland08.yaml")
    random.shuffle(propositions)
    print "Read %d propositions" % len(propositions)
