import numpy as np

def recItem(model, user_train, num_user_to_search, item_idx, k, args):

    train = user_train
    users = range(1, num_user_to_search)

    item_preds = []
    
    for u in users:

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1

        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: 
                break

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])

        predictions_origin = predictions

        predictions = predictions_origin.argsort().argsort()[0].cpu().detach().numpy()

        rank = np.argmax(predictions)

        for i in reversed(train[u]):
            if i != item_idx[rank]:
                item_preds.append(i)

    item_preds = list(set(item_preds))[0:k]

    return item_preds
    