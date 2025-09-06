import tensorflow as tf

def train_model(strategy, model, optimizer, loss_fn, dataset, rank):
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

    print(f"RANK {rank}: Reached barrier")
    tf.distribute.experimental_collective_ops.barrier()

    for epoch in range(5):
        dist_dataset = strategy.experimental_distribute_dataset(dataset)
        total_loss, num_batches = 0.0, 0

        for batch in dist_dataset:
            def step_fn(inputs):
                x, y = inputs
                with tf.GradientTape() as tape:
                    preds = model(x, training=True)
                    loss = loss_fn(y, preds)
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                return loss

            per_replica_losses = strategy.run(step_fn, args=(batch,))
            batch_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
            total_loss += batch_loss
            num_batches += 1

        if rank == 0:
            print(f"[Epoch {epoch}] Loss: {total_loss / num_batches:.4f}")
