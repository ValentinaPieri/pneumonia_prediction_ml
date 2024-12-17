from .hash_chain_digest import HashChainDigest
from .message import Message
from .message_type import MessageType

class Entry:
    def __init__(self, replica_id: int, view: int, sequence_number: int, hcd: HashChainDigest):
        """
        Initialize an Entry object.

        This constructor initializes an Entry object with the id of the replica that executed the request operation, the view number, the sequence number of the request, and the hcd block for sequence number n.

        Args:
            replica_id (int): The id of the replica that executed the request operation.
            view (int): The view number.
            sequence_number (int): The sequence number of the request.
            hcd (HashChainBlock): The hash chain digest after n operations.

        Attributes:
            replica_id (int): The id of the replica that executed the request operation.
            view (int): The view number.
            sequence_number (int): The sequence number of the request.
            hcd (HashChainBlock): The hash chain digest after n operations."""
        self.replica_id = replica_id
        self.view = view
        self.sequence_number = sequence_number
        self.hcd = hcd

    def get_replica_id(self) -> int:
        """
        Get the id of the replica that executed the request operation.

        Returns:
            int: The id of the replica that executed the request operation."""
        return self.replica_id

    def get_view(self) -> int:
        """
        Get the view number.

        Returns:
            int: The view number."""
        return self.view

    def get_sequence_number(self) -> int:
        """
        Get the sequence number of the request.

        Returns:
            int: The sequence number of the request."""
        return self.sequence_number

    def get_hcd(self) -> HashChainDigest:
        """
        Get the hash chain digest after n operations.

        Returns:
            HashChainBlock: The hash chain digest after n operations."""
        return self.hcd

    def to_string(self):
        """
        Convert the Entry object to a string.

        Returns:
            str: A string representation of the Entry object."""
        return f"Replica ID: {self.replica_id}, View: {self.view}, Sequence Number: {self.sequence_number}, HCD: {self.hcd}"

class ReplyMessage(Message):
    def __init__(self, client_id: int, timestamp: float, result: str, replica_id: int, view: int, sequence_number: int, hcd: HashChainDigest):
        """
        Initialize a Reply message.

        This constructor initializes a message of type `REPLY` with a given client ID, timestamp, result, and entry.

        Args:
            client_id (int): The ID of the client that issued the request.
            timestamp (float): The timestamp of the request. Calculated as the time when the request was issued.
            result (str): The result obtained from executing the request operation.
            replica_id (int): The ID of the replica that executed the request operation.
            view (int): The view number.
            sequence_number (int): The sequence number of the request.
            hcd (HashChainDigest): The hash chain digest after n operations.

        Attributes:
            client_id (int): The ID of the client that issued the request.
            timestamp (float): The timestamp of the request.
            result (str): The result obtained from executing the request operation.
            entry (Entry): formed by the id of the replica that executed the request operation, the view number, the sequence number of the request, and the hcd block for sequence number n. The entry is signed by the replica that executed the request operation and used by the client to update its version vector.
        """
        super().__init__(MessageType.REPLY)
        self.client_id = client_id
        self.timestamp = timestamp
        self.result = result
        self.entry = Entry(replica_id, view, sequence_number, hcd)

    def get_client_id(self) -> int:
        """
        Get the ID of the client that issued the request.

        Returns:
            int: The ID of the client that issued the request."""
        return self.client_id

    def get_timestamp(self) -> float:
        """
        Get the timestamp of the request.

        Returns:
            float: The timestamp of the request."""
        return self.timestamp

    def get_result(self) -> str:
        """
        Get the result obtained from executing the request operation.

        Returns:
            str: The result obtained from executing the request operation."""
        return self.result

    def get_entry(self) -> Entry:
        """
        Get the entry of the reply message.

        Returns:
            Entry: The entry of the reply message."""
        return self.entry

    def to_string(self):
        """
        Convert the Reply message to a string.

        Returns:
            str: The Reply message as a string."""
        return super().to_string() + f"Client ID: {self.client_id}, Timestamp: {self.timestamp}, Result: {self.result}, Entry: {self.entry.to_string()}"