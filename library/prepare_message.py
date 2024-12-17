from .request_message import RequestMessage
from .message import Message
from .message_type import MessageType

class PrepareMessage(Message):
    def __init__(self, replica_id: int, view: int, sequence_number: int, request_msg: RequestMessage):
        """
        Initialize a Prepare message.

        This constructor initializes a message of type `PREPARE` with a given sequence number, hash chain digest and replica ID.

        Args:
            replica_id (int): The ID of the replica that issued the prepare message.
            view (int): The view number.
            sequence_number (int): The sequence number assigned by the primary replica.
            request_msg (RequestMessage): The request message.

        Attributes:
            replica_id (int): The ID of the replica that issued the prepare message.
            request_msg (RequestMessage): The request message.
        """
        super().__init__(MessageType.PREPARE, view, sequence_number)
        self.replica_id = replica_id
        self.request_msg = request_msg

    def get_replica_id(self) -> int:
        """
        Get the ID of the replica that issued the prepare message.

        Returns:
            int: The ID of the replica that issued the prepare message.
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
        Get the sequence number assigned by the primary replica.

        Returns:
            int: The sequence number assigned by the primary replica.
        """
        return self.sequence_number

    def get_request_msg(self) -> RequestMessage:
        """
        Get the request message.

        Returns:
            RequestMessage: The request message.
        """
        return self.request_msg

    def to_string(self):
        """
        Convert the message to a string.

        Returns:
            str: The string representation of the message.
        """
        return super().to_string() + f"Replica ID: {self.replica_id}, Hashed Request Message: {self.request_msg}"