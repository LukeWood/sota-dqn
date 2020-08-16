from .basic_replay_memory import BasicReplayMemory


class TestBasicMemory:
    def test_save(self):
        mem = BasicReplayMemory()
        mem.save("Experience")
        assert ["Experience"] == mem.sample(1)

    def test_max_size(self):
        mem = BasicReplayMemory(size=1)
        mem.save("Hello")
        assert ["Hello"] == mem.sample(1)

        assert len(mem) == 1
        mem.save("Experience")
        assert ["Hello"] != mem.sample(1)

        assert len(mem) == 1
        mem.save("Final")
        assert ["Final"] == mem.sample(1)
