from disjoint_set import DisjointSet

class DisjointSetExtra():
    def __init__(self):
        self.ds = DisjointSet()
        self.ds_counter = dict()

# one of the two nodes doesnot exist
    def add(self,node1,node2):
        if node1 in self.ds:
            self.ds.union(node2,node1)
            self.ds_counter[self.find(node1)] += 1
        elif node2 in self.ds:
            self.ds.union(node1,node2)
            self.ds_counter[self.find(node2)] += 1
        else:
            self.ds.union(node2,node1)
            self.ds_counter[self.find(node2)] = 2


    def find(self,node):
        return self.ds.find(node)

    def count(self,node):
        return self.ds_counter[self.find(node)]

    # added
    def connect(self,node1,node2):
        name1 = self.find(node1)
        name2 = self.find(node2)
        if self.ds_counter[name1] < self.ds_counter[name2]:
            self.ds_counter[name2]  = self.ds_counter[name2]+self.ds_counter[name1]
            self.union(node1,node2)
            del self.ds_counter[name1]
        else:
            self.ds_counter[name1]  = self.ds_counter[name2]+self.ds_counter[name1]
            self.union(node2,node1)
            del self.ds_counter[name2]
    # added
    def exists(self,node):
        return node in self.ds

    def union(self,node1,node2):
        self.ds.union(node1,node2)

    def connected(self,node1,node2):
        return self.ds.connected(node1,node2)
