# Train loop for contrastive pretraining
def train_pretrain(args, model, device, train_loader, optimizer, epoch, logger):
  # have to remember to put your model in training mode!
    model.train()
    # logs loss per-batch to the console
    # stats is a range object from zero to len(train_loader)
    stats = trange(len(train_loader))
    indexer = epoch * len(train_loader)
    for batch_idx, (anc, pos, neg) in zip(stats, train_loader):
       # three items: anchor image, positive pair, and negative pair
        anc, pos, neg = anc.to(device), pos.to(device), neg.to(device)
        optimizer.zero_grad()
        out_anc = model(anc)
        out_pos = model(pos)
        out_neg = model(neg)
        # other losses that might be fun to try are
        # info NCE https://github.com/RElbers/info-nce-pytorch
        # or a cosine-based triplet loss https://pytorch.org/docs/stable/generated/torch.nn.CosineEmbeddingLoss.html
        loss = F.triplet_margin_loss(out_anc, out_pos, out_neg, margin=0.05)
        # margin is a tuneable hyperparameter
        # margin taken as mean of two previous triplet loss papers
        # here (margin = 0.3) https://github.com/omipan/camera_traps_self_supervised/blob/6a88c2abf326daca4c3b69b1f62c6c05b7b8a7e0/losses.py#L31
        # and here (margin = 0.1) https://github.com/ermongroup/tile2vec/blob/b24ee8d046b3a2e5233d6ac9fa62380ce8cab031/src/resnet.py#L162
        loss.backward()
        optimizer.step()
        # save train loss to tqdm bar
        stats.set_description(f'epoch {epoch}')
        stats.set_postfix(loss=loss.item())
        logger[batch_idx+indexer] = loss.item()
    stats.close()
    return logger