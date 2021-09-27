# -*- coding: utf-8 -*-
import os
import re
import sys
import csv
import rich
import logging
import argparse
import numpy as np
import pandas as pd
import py_entitymatching as em

from matching import *


def main(args):
	conf_fn = os.path.join(args.data_dir, args.conf_fn)
    jour_fn = os.path.join(args.data_dir, args.jour_fn)
    test_fn = os.path.join(args.data_dir, args.test_fn)

    CONF_INDEX = 'id'
    CONF = ['id', 'year', 'bow', 'authors', 'authors_first_last', 'first_author_first', 'first_author_last']
    CONF = em.read_csv_metadata(conf_fn, key=CONF_INDEX)

    JOUR_INDEX = 'id'
    JOUR = ['id', 'year', 'bow', 'authors', 'authors_first_last']
    JOUR = em.read_csv_metadata(jour_fn, key=JOUR_INDEX)

	test_meta_data = em.read_csv_metadata(test_fn, 
		key='_id', ltable=JOUR, rtable=CONF, fk_ltable='ltable_'+JOUR_INDEX, fk_rtable='rtable_'+CONF_INDEX)

	test_set = em.read_csv_metadata(
        test_fn, 
        key='_id',
        ltable=JOUR, 
        rtable=CONF,
        fk_ltable='ltable_'+JOUR_INDEX, 
        fk_rtable='rtable_'+CONF_INDEX
    )

	features_fn = os.path.join(args.data_dir, args.features_fn)
    if os.path.exists(features_fn):
        test_feature_vectors = pd.read_csv(features_fn)
    else:
        test_feature_vectors = em.extract_feature_vecs(
            test_set, feature_table=feature_meta_data, attrs_after='gold_label')
        test_feature_vectors.to_csv(features_fn, index=False)

	# Impute feature vectors with the mean of the column values.
	em.set_key(test_feature_vectors, '_id') # key of the metadata
	em.set_fk_ltable(test_feature_vectors, 'ltable_id') #foreign key to left table
	em.set_fk_rtable(test_feature_vectors, 'rtable_id') #foreign key to right table
	em.set_ltable(test_feature_vectors, JOUR) #Sets the ltable for a DataFrame in the catalog
	em.set_rtable(test_feature_vectors, CONF) #Sets the rtable for a DataFrame in the catalog

	test_feature_vectors = em.impute_table(test_feature_vectors, exclude_attrs=['_id', 'gold_label'], missing_val=np.nan, strategy='mean')

	predictions = rf.predict(table=test_feature_vectors, exclude_attrs=['_id',  'gold_label'], append=True, target_attr='predicted', inplace=False)
	predictions.to_csv("{}/predictions_210607.csv".format(path_to_csv_dir), index=False)

	eval_summary = em.eval_matches(predictions, 'gold_label', 'predicted')
	em.print_eval_summary(eval_summary)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help="Path to dataset directory", required=True)
    parser.add_argument('--conf_fn', type=str, help="Conference metadata filename", required=True)
    parser.add_argument('--jour_fn', type=str, help="Journal metadata filename", required=True)
    parser.add_argument('--test_fn', type=str, help="Name of the test file", required=True)
    parser.add_argument('--features_fn', type=str, help="Name of the features file", required=True)
    parser.add_argument('--model_path', type=str, help="Path to the matching model", required=True)
    args = parser.parse_args()
    main(args)

