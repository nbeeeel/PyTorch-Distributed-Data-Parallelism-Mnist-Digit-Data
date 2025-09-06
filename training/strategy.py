import tensorflow as tf
import socket, time

def init_strategy(rank, master_addr, master_port):
    if rank != 0:
        try:
            socket.create_connection((master_addr, int(master_port)), timeout=10)
            print(f"RANK {rank}: Connected to {master_addr}:{master_port}")
        except socket.error as e:
            print(f"RANK {rank}: Failed to connect: {e}")

    for attempt in range(3):
        try:
            print(f"RANK {rank}: Attempt {attempt+1} to init strategy")
            return tf.distribute.MultiWorkerMirroredStrategy(
                communication_options=tf.distribute.experimental.CommunicationOptions(
                    implementation=tf.distribute.experimental.CommunicationImplementation.NCCL
                )
            )
        except Exception as e:
            print(f"RANK {rank}: Strategy init failed: {e}")
            if attempt < 2: time.sleep(5)
            else: raise
