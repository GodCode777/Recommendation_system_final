import numpy as np


def get_content_based_recommendations(items, item_vectors, items_id, k):
    """
    This function will return contect based recommended items.
    items: items is a dictionary with keys as item_id and values as 'pos' or 'neg'
    item_vectors: array of item vectors
    items_id: list of item id according to the item_vectors array
    k: k-top most items
    """
    user_profile = np.zeros(100)
 
    pos_items_interacted = [item for item in items.keys() if items[item] == 'pos']
    neg_items_interacted = [item for item in items.keys() if items[item] == 'neg']

    arg_items = np.where(np.isin(items_id,pos_items_interacted))[0]
    user_profile += np.sum(item_vectors[arg_items], axis = 0)
    arg_items = np.where(np.isin(items_id,neg_items_interacted))[0]
    user_profile -= np.sum(item_vectors[arg_items], axis = 0)
    user_profile = user_profile / (len(pos_items_interacted) + len(neg_items_interacted))

    rec_items = items_id[np.argsort(np.sqrt(np.sum(np.square(item_vectors - user_profile), axis = 1)))[:k]]

    return rec_items


def get_matrix_factorized_recommendations(b, user_keys, user_id, k):
    if user_id in user_keys:
        user_indx = b['user_dict'][user_id]
        temp = np.dot(b['p'][user_indx], b['q']) + b['bi'] + b['bu'][user_indx] + b['u']
        rec_indx = (-np.array(temp)).argsort()[:k]
        rec_items = np.array(list(b['item_dict'].keys()))[np.array(list(b['item_dict'].values()))[rec_indx]]

        return rec_items
    else:
        return []
