from .message import Message
from .message_type import MessageType

class RequestMessage(Message):
    def __init__(self, operation : str, timestamp : float, client_id : int, current_system_state : str):
        """
        Initialize a Request message.

        This constructor initializes a message of type `REQUEST` with a given operation, timestamp, client ID, and current system state.

        Args:
            operation (str): The operation to be performed.
            timestamp (float): The timestamp of the request. Calculated as the time when the request was issued.
            client_id (int): The ID of the client that issued the request.
            current_system_state (str): Last known state of the system by the client that issued the request.

        Attributes:
            operation (str): The operation to be performed.
            timestamp (float): The timestamp of the request. Calculated as the time when the request was issued.
            client_id (int): The ID of the client that issued the request.
            current_system_state (str): Last known state of the system by the client that issued the request.
        """
        super().__init__(MessageType.REQUEST)
        self.operation = operation
        self.timestamp = timestamp
        self.client_id = client_id
        self.current_system_state = current_system_state

    def get_operation(self) -> str:
        """
        Get the operation to be performed.

        Returns:
            str: The operation to be performed.
        """
        return self.operation

    def get_timestamp(self) -> float:
        """
        Get the timestamp of the request.

        Returns:
            float: The timestamp of the request.
        """
        return self.timestamp

    def get_client_id(self) -> int:
        """
        Get the ID of the client that issued the request.

        Returns:
            int: The ID of the client that issued the request.
        """
        return self.client_id

    def get_current_system_state(self) -> str:
        """
        Get the last known state of the system by the client that issued the request.

        Returns:
            str: The last known state of the system by the client that issued the request.
        """
        return self.current_system_state

    def to_string(self):
        """
        Convert the message to a string.

        Returns:
            str: The string representation of the message.
        """
        return super().to_string() + f"Client ID: {self.client_id}, Operation: {self.operation}, Timestamp: {self.timestamp}, Current System State: {self.current_system_state}"