from .version_vector import VersionVectorEntry
from .message import Message
from .message_type import MessageType

class ViewChangeMessage(Message):
    def __init__(self, new_view: int, replica_id: int, vv_replica_id: int, vv_view: int, vv_sequence_number: int, vv_hcd, P: list):
        """
        Initialize a View Change message.

        This constructor initializes a message of type `VIEW_CHANGE` with a given new view, version vector entry, and P.

        Args:
            new_view (int): The new view to be applied.
            version_vector_entry (VersionVectorEntry): The version vector entry for r's last committed operation.
            P (list): The set of sets P_m for each pre-prepared message m with sequence number higher than n. Each P_m contains the pre-prepare message for m and 2f corresponding matching prepare messages.

        Attributes:
            new_view (int): The new view to be applied.
            version_vector_entry (VersionVectorEntry): The version vector entry for r's last committed operation.
            P (list): The set of sets P_m for each pre-prepared message m with sequence number higher than n. Each P_m contains the pre-prepare message for m and 2f corresponding matching prepare messages.
        """
        super().__init__(MessageType.VIEW_CHANGE, new_view)
        self.replica_id = replica_id
        entry = VersionVectorEntry(vv_replica_id, vv_view, vv_sequence_number, vv_hcd)
        self.version_vector_entry = entry
        self.P = P

    def get_version_vector_entry(self) -> VersionVectorEntry:
        """
        Get the version vector entry for r's last committed operation.

        Returns:
            VersionVectorEntry: The version vector entry for r's last committed operation.
        """
        return self.version_vector_entry

    def get_P(self):
        """
        Get the set of sets P_m for each pre-prepared message m with sequence number higher than n. Each P_m contains the pre-prepare message for m and 2f corresponding matching prepare messages.

        Returns:
            list: The set of sets P_m for each pre-prepared message m with sequence number higher than n. Each P_m contains the pre-prepare message for m and 2f corresponding matching prepare messages.
        """
        return self.P

    def get_replica_id(self) -> int:
        """
        Get the ID of the replica that issued the view change message.

        Returns:
            int: The ID of the replica that issued the view change message.
        """
        return self.replica_id

    def to_string(self):
        """
        Convert the view change message to a string.

        Returns:
            str: The string representation of the view change message.
        """
        return super().to_string() + f"New View: {self.get_view()}, Version Vector Entry: {self.version_vector_entry}, P: {self.P}"