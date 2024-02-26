#!/usr/bin/env python3
from argparse import ArgumentParser, FileType
from importlib import import_module
from itertools import count
import os

import h5py
import json
import numpy as np
from sklearn.metrics import average_precision_score
import tensorflow as tf

import common
import loss


parser = ArgumentParser(description='Evaluate a ReID embedding.')

parser.add_argument(
    '--excluder', required=True, choices=('market1501', 'diagonal','cuhk03','duke'),
    help='Excluder function to mask certain matches. Especially for multi-'
         'camera datasets, one often excludes pictures of the query person from'
         ' the gallery if it is taken from the same camera. The `diagonal`'
         ' excluder should be used if this is *not* required.')

parser.add_argument(
    '--query_dataset', required=True,
    help='Path to the query dataset csv file.')

parser.add_argument(
    '--query_embeddings', required=True,
    help='Path to the h5 file containing the query embeddings.')

parser.add_argument(
    '--gallery_dataset', required=True,
    help='Path to the gallery dataset csv file.')

parser.add_argument(
    '--gallery_embeddings', required=True,
    help='Path to the h5 file containing the query embeddings.')

parser.add_argument(
    '--metric', default='euclidean', required=True, choices=loss.cdist.supported_metrics,
    help='Which metric to use for the distance between embeddings.')

parser.add_argument(
    '--filename', type=FileType('w'),
    help='Optional name of the json file to store the results in.')

parser.add_argument(
    '--batch_size', default=256, type=common.positive_int,
    help='Batch size used during evaluation, adapt based on your memory usage.')


def main():
    # Verify that parameters are set correctly.
    args = parser.parse_args()

    # Load the query and gallery data from the CSV files.
    query_pids, query_fids = common.load_dataset(args.query_dataset, None)
    gallery_pids, gallery_fids = common.load_dataset(args.gallery_dataset, None)

    # Load the two datasets fully into memory.
    with h5py.File(args.query_embeddings, 'r') as f_query:
        query_embs = np.array(f_query['emb'])
        query_embs_dp = np.array(f_query['emb'])

    with h5py.File(args.gallery_embeddings, 'r') as f_gallery:
        gallery_embs = np.array(f_gallery['emb'])
        gallery_embs_dp = np.array(f_gallery['emb'])

    # Just a quick sanity check that both have the same embedding dimension!
    query_dim = query_embs.shape[1]
    gallery_dim = gallery_embs.shape[1]
    if query_dim != gallery_dim:
        raise ValueError('Shape mismatch between query ({}) and gallery ({}) '
                         'dimension'.format(query_dim, gallery_dim))

    # Setup the dataset specific matching function
    excluder = import_module('excluders.' + args.excluder).Excluder(gallery_fids)
    
    query_embs_dp[0,1]=np.random.normal(0.021068161,2.312213)
    query_embs_dp=np.exp(1) * query_embs_dp +0.01
    gallery_embs_dp[0,1]=np.random.normal(0,2.3505316)
    gallery_embs_dp=np.exp(1) * gallery_embs_dp +0.01
    
    def mse(actual, pred): 
        actual, pred = np.array(actual), np.array(pred)
        return np.max(np.square(np.subtract(actual,pred)).mean())

    D0 = query_embs + np.random.normal(0,1.3787516)
    D1 = query_embs_dp + np.random.normal(0,1.3787516)

    D2 = gallery_embs + np.random.normal(0,1.3921114)
    D3 = gallery_embs_dp + np.random.normal(0,1.3921114)
    
    #print(D0,D1)
    #sens1 = l1(query_embs,query_embs_dp)
    sens2 = mse(D0,D1)
    print(sens2)

    # We go through the queries in batches, but we always need the whole gallery
    batch_pids, batch_fids, batch_embs = tf.data.Dataset.from_tensor_slices(
        (query_pids, query_fids,D1)
    ).batch(args.batch_size).make_one_shot_iterator().get_next()

    batch_distances = loss.cdist(batch_embs, D3, metric=args.metric)
    id_batch_distances = loss.cdist(batch_embs, gallery_embs, 'ce')


    # Loop over the query embeddings and compute their APs and the CMC curve.
    aps = []
    cmc = np.zeros(len(gallery_pids), dtype=np.int32)
    acc = np.zeros(len(gallery_pids), dtype=np.int32)

    with tf.Session() as sess:
        for start_idx in count(step=args.batch_size):
            try:
                # Compute distance to all gallery embeddings
                distances, dist, pids, fids = sess.run([
                    batch_distances, id_batch_distances, batch_pids, batch_fids])
                print('\rEvaluating batch {}-{}/{}'.format(
                        start_idx, start_idx + len(fids), len(query_fids)),
                      flush=True, end='')
            except tf.errors.OutOfRangeError:
                print()  # Done!
                break

            # Convert the array of objects back to array of strings
            #print(pids)
            pids, fids = np.array(pids, '|U'), np.array(fids, '|U')
            print(pids)

            # Compute the pid matches
            pid_matches = gallery_pids[None] == pids[:,None]

            # Get a mask indicating True for those gallery entries that should
            # be ignored for whatever reason (same camera, junk, ...) and
            # exclude those in a way that doesn't affect CMC and mAP.
            mask = excluder(fids)
            distances[mask] = np.inf
            pid_matches[mask] = False

            # Keep track of statistics. Invert distances to scores using any
            # arbitrary inversion, as long as it's monotonic and well-behaved,
            # it won't change anything.
            scores = 1 / (1 + distances)
            for i in range(len(distances)):
                ap = average_precision_score(pid_matches[i], scores[i])

                if np.isnan(ap):
                    print()
                    print("WARNING: encountered an AP of NaN!")
                    print("This usually means a person only appears once.")
                    print("In this case, it's because of {}.".format(fids[i]))
                    print("I'm excluding this person from eval and carrying on.")
                    print()
                    continue

                aps.append(ap)
                # Find the first true match and increment the cmc data from there on.
                k = np.where(pid_matches[i, np.argsort(distances[i])])[0][0]
                cmc[k:] += 1

                for j in range(len(dist)):
                    ids = np.where(pid_matches[j, np.argsort(dist[j])])[0][0]
                    acc[ids:] += 1

    # Compute the actual cmc and mAP values
    cmc = cmc / len(query_pids)
    acc = acc / len(query_pids)
    mean_ap = np.mean(aps)

    # Save important data
    if args.filename is not None:
        json.dump({'mAP': mean_ap, 'CMC': list(cmc), 'aps': list(aps)}, args.filename)

    # Print out a short summary.
    print('mAP: {:.2%} | top-1: {:.2%} top-2: {:.2%} | top-5: {:.2%} | top-10: {:.2%} | acc: {:.2%}'.format(
        mean_ap, cmc[0], cmc[1], cmc[4], cmc[9], acc[0]))

if __name__ == '__main__':
    main()