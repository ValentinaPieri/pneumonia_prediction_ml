from .hash_chain_digest import HashChainDigest

class VersionVectorEntry:
    """
    Represents an entry in the version vector.
    """
    def __init__(self, replica_id: int, view: int, sequence_number: int, hcd: HashChainDigest):
        """
        Initialize an Entry object.

        An entry in the version vector contains the replica id, the view number, the sequence number, and the hash chain digest after n operations.

        Args:
            replica_id (int): The id of the replica.
            view (int): The view number.
            sequence_number (int): The sequence number.
            hcd (HashChainDigest): The hash chain digest after n operations.

        Attributes:
            replica_id (int): The id of the replica.
            view (int): The view number.
            sequence_number (int): The sequence number.
            hcd (HashChainDigest): The hash chain digest after n operations.
        """
        self.replica_id = replica_id
        self.view = view
        self.sequence_number = sequence_number
        self.hcd = hcd # HCD^n (hash chain digest after n operations)

    def get_replica_id(self) -> int:
        """
        Get the id of the replica.

        Returns:
            int: The id of the replica.
        """
        return self.replica_id

    def get_view(self) -> int:
        """
        Get the view number.

        Returns:
            int: The view number.
        """
        return self.view

    def get_sequence_number(self) -> int:
        """
        Get the sequence number.

        Returns:
            int: The sequence number.
        """
        return self.sequence_number

    def get_hcd(self):
        """
        Get the hash chain digest after n operations.

        Returns:
            HashChainDigest: The hash chain digest after n operations.
        """
        return self.hcd

class VersionVector:
    """Represents a version vector. The vector is a dictionary where keys are replica/client IDs, and values are VersionVectorEntry objects."""
    def __init__(self):
        """
        Initialize a VersionVector object.

        Attributes:
            vector (dict): A dictionary where keys are replica/client IDs, and values are VersionVectorEntry objects."""
        self.vector = {}

    def update_entry(self, vector_id: int, entry_id: int, view: int, sequence_number: int, hcd: HashChainDigest):
        """
        Updates the entry for the specified replica/client with the given view, sequence number, and hash chain digest. If the entry does not exist, it is created.

        Args:
            vector_id (int): The id of the replica/client.
            entry_id (int): The id of the entry.
            view (int): The view number.
            sequence_number (int): The sequence number.
            hcd (HashChainDigest): The hash chain digest after n operations.
        """
        if vector_id not in self.vector:
            self.vector[vector_id] = []
        self.vector[vector_id].append(VersionVectorEntry(entry_id, view, sequence_number, hcd))

    def get_entries(self, replica_id: int):
        """Get the entries for the specified replica.

        Args:
            replica_id (int): The id of the replica whose entries are being queried.

        Returns:
            list: The list of entries for the specified replica.
        """
        return self.vector[replica_id]

    def get_entry(self, vector_replica_id: int, entry_replica_id: int) -> VersionVectorEntry:
        """
        Get the entry for the specified replica in the specified vector.

        Args:
            vector_replica_id (int): The id of the replica whose vector is being queried.
            entry_replica_id (int): The id of the replica whose entry in the specified vector is being queried.

        Returns:
            VersionVectorEntry: The entry for the specified replica in the specified vector
        """
        return self.get_entries(vector_replica_id)[entry_replica_id]

    def to_string(self):
        """
        Convert the version vector to a string. The string contains the replica id, view number, sequence number, and hash chain digest for each entry in the vector.

        Returns:
            str: A string representation of the version vector.
        """
        vector_str = ""
        for replica, entries in self.vector.items():
            for entry in entries:
                vector_str += f"⟨{entry.replica}, {entry.view}, {entry.sequence_number}, {entry.hcd}⟩ "
        return vector_str.strip()

    def is_empty(self):
        """Check if the version vector is empty.

        Returns:
            bool: True if the version vector is empty, False otherwise.
        """
        return not bool(self.vector)

    def get_current_system_state(self, f) -> VersionVectorEntry:
        """Returns the current system state based on the version vector entries.

        Args:
            f (int): The number of faulty replicas tolerated by the system.

        Returns:
            VersionVectorEntry: The current system state.
        """
        counter = {}
        for entries in self.vector.values():
            for entry in entries:
                key = (str(entry.get_sequence_number()), entry.get_hcd())
                if key not in counter:
                    counter[key] = []
                counter[key].append(entry)

        for key, entries in counter.items():
            if len(entries) >= ((2 * f) + 1):
                return min(entries, key=lambda entry: entry.get_replica_id())
        return