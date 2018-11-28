import uuid


class LessSimpleNode:
    def __init__(self, id, data=None):
        self._id = id
        self._hash = uuid.uuid4()

        for k, v in data.items():
            setattr(self, k, v)

    def id(self):
        return self._id

    def f(self, x):
        try:
            return self.__dict__['function'](x)
        except KeyError:
            print('No function associated with this node')

    def __hash__(self):
        return hash(self._id)

    def __repr__(self):
        return str(self._id)

    def __eq__(self, other):
        return self._id == other._id

    def __getattr__(self, attr):
        return 'Node %d has no attribute %s' % (self._id, attr)
