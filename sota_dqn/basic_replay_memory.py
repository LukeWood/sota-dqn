import random


class BasicReplayMemory:
    def __init__(self, size=2000):
        self.size = size
        self.memory = []
        self.index = 0

    def save(self, frame):
        if len(self.memory) < self.size:
            self.memory.append(frame)
        else:
            self.memory[self.index % self.size] = frame
            self.index = self.index + 1

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
