import socket
import threading

from .client import Client
from .version_vector import VersionVector
from .reply_cache import ReplyCache
from .hash_chain_digest import HashChainDigest
from .message_type import MessageType
from .operations_dict import OperationsDict
from .checkpoint_message import CheckpointMessage
from .request_message import RequestMessage
from .preprepare_message import PrePrepareMessage
from .prepare_message import PrepareMessage
from .commit_message import CommitMessage
from .view_change_message import ViewChangeMessage
from .reply_message import ReplyMessage
from .new_view_message import NewViewMessage
from threading import Lock, Timer
from typing import List

VIEW_TIMEOUT = 10
CHECKPOINT_INTERVAL = 100
I = 128

class Replica():
    """
    A class representing a replica in the BFT2F protocol. The replica is responsible for processing client requests, multicasting messages to other replicas, and maintaining the state of the system.
    """
    _id_counter = 0
    def __init__(self, host, port, f, view = 0, clients: List['Client'] = [], replicas: List['Replica'] = []):
        """
        Initialize a Replica object and start a thread to listen for incoming messages.

        Args:
            host (str): The IP address of the replica.
            port (int): The port number of the replica.
            f (int): The number of faulty replicas.
            view (int): The view number.
            clients (List[Client]): A list of clients connected to the replica.
            replicas (List[Replica]): A list of replicas in the system.

        Attributes:
            host (str): The IP address of the replica.
            port (int): The port number of the replica.
            lock (Lock): A lock to ensure thread safety.
            f (int): The number of faulty replicas. The system can tolerate f faults.
            replica_id (int): The ID of the replica.
            view (int): The view number.
            reply_cache (ReplyCache): A cache to store replies to client requests.
            version_vector (VersionVector): A vector to store the state of each replica.
            hcd (HashChainDigest): The hash chain digest for the replica.
            clients (List[Client]): A list of clients connected to the replica.
            replicas (List[Replica]): A list of replicas in the system.
            primary (bool): A boolean indicating whether the replica is the primary replica. Calculated based on the view number, the replica ID and the f value.
            accepted_requests (List[RequestMessage]): A list of accepted client requests.
            received_pre_prepare_messages (List[PrePrepareMessage]): A list of received pre-prepare messages.
            received_prepare_messages (List[PrepareMessage]): A list of received prepare messages.
            received_commit_messages (List[CommitMessage]): A list of received commit messages.
            received_view_change_messages (List[ViewChangeMessage]): A list of received view change messages.
            timer (Timer): A timer to trigger view changes.
            """
        self.host = host
        self.port = port
        self.lock = Lock()
        self.f = f
        self.replica_id = Replica._id_counter
        Replica._id_counter += 1
        self.view = view
        self.reply_cache = ReplyCache()
        self.version_vector = VersionVector()
        self.hcd = HashChainDigest()
        self.clients = clients
        self.replicas = replicas
        self.primary = True if self.view % (3*(self.f + 1)) == self.replica_id else False
        self.accepted_requests = []
        self.received_pre_prepare_messages = []
        self.received_prepare_messages = []
        self.received_commit_messages = []
        self.received_view_change_messages = []
        self.pending_checkpoints = {}
        self.timer = None

        self.listen()

    def get_host(self):
        """
        Get the IP address of the replica.

        Returns:
            str: The IP address of the replica.
        """
        return self.host

    def get_port(self):
        """
        Get the port number of the replica.

        Returns:
            int: The port number of the replica."""
        return self.port

    def get_replica_id(self) -> int:
        """
        Get the ID of the replica.

        Returns:
            int: The ID of the replica.
        """
        return self.replica_id

    def get_view(self) -> int:
        """
        Get the view number.

        Returns:
            int: The view number.
        """
        return self.view

    def get_reply_cache(self) -> ReplyCache:
        """
        Get the reply cache.

        Returns:
            ReplyCache: The reply cache.
        """
        return self.reply_cache

    def get_version_vector(self) -> VersionVector:
        """
        Get the version vector.

        Returns:
            VersionVector: The version vector.
        """
        return self.version_vector

    def get_hcd(self) -> HashChainDigest:
        """
        Get the hash chain digest.

        Returns:
            HashChainDigest: The hash chain digest.
        """
        return self.hcd

    def get_fault_tolerance(self) -> int:
        """
        Get the fault tolerance value.

        Returns:
            int: The fault tolerance value.
        """
        return self.f

    def get_latest_reply(self, client_id) -> ReplyMessage:
        """
        Get the latest reply for a given client ID.

        Args:
            client_id (int): The ID of the client.

        Returns:
            ReplyMessage: The latest reply for the client.
        """
        return self.reply_cache.get_reply(client_id)

    def get_replicas(self):
        """
        Get the list of replicas in the system.

        Returns:
            List[Replica]: A list of replicas in the system.
        """
        return self.replicas

    def is_primary(self):
        """
        Check if the replica is the primary replica.

        Returns:
            bool: True if the replica is the primary replica, False otherwise.
        """
        return self.primary

    def add_replica(self, replica: 'Replica'):
        """
        Add a replica to the list of replicas in the system.
        """
        if replica not in self.replicas:
            self.replicas.append(replica)

    def update_reply_cache(self, client_id, reply):
        """
        Update the reply cache with the latest reply for a given client ID.

        Args:
            client_id (int): The ID of the client.
            reply (ReplyMessage): The latest reply for the client.
        """
        self.reply_cache.update_cache(client_id, reply)

    def start_timer(self):
        """
        Start the timer to trigger view changes.
        """
        if self.timer is None:
            self.timer = Timer(VIEW_TIMEOUT, self.send_view_change, args=[])  # Corrected this line
            self.timer.start()

    def listen(self):
        """
        Start a thread to listen for incoming messages.
        """
        listen_thread = threading.Thread(target=self._listen_thread)
        listen_thread.daemon = False
        listen_thread.start()

    def _listen_thread(self):
        """
        Listen for incoming messages and process them accordingly, based on the message type.

        Throws:
            Exception: If the message type is invalid.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((self.host, self.port))

            if self.host.startswith('224.') or self.host.startswith('225.') or self.host.startswith('226.') or self.host.startswith('227.'):
                mreq = socket.inet_aton(self.host) + socket.inet_aton('0.0.0.0')
                s.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

            while True:
                data, addr = s.recvfrom(1024)
                msg = data.decode()
                if not msg:
                    break
                msg = self.deserialize_message(msg)
                if isinstance(msg, Exception):
                    continue
                if msg == 'MessageType.PRIMARY_NOT_DOMINANT':
                    if (not self.is_primary()):
                        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                            s.connect(self.find_primary_replica().get_host(), self.find_primary_replica().get_port())
                            sequence_number_1 = self.find_primary_replica().get_version_vector().get_entry(self.find_primary_replica().get_replica_id(), self.find_primary_replica().get_replica_id()).get_sequence_number()
                            sequence_number_2 = self.get_version_vector().get_entry(self.get_replica_id(), self.get_replica_id()).get_sequence_number()
                            dict = {}
                            while (sequence_number_2 <= sequence_number_1):
                                list = []
                                total = 0
                                sequence_number_2 += 1
                                for commit in self.received_commit_messages:
                                    if commit.get_sequence_number() == sequence_number_2:
                                        total += 1
                                        list.append(commit)
                                if (total >= (2 * self.f) + 1):
                                    dict = {sequence_number_2: list}
                            messages_dict = OperationsDict(dict)
                            s.send(messages_dict.encode())
                match msg.get_message_type():
                    case MessageType.REQUEST:
                        self.receive_request(msg)
                    case MessageType.PRE_PREPARE:
                        self.receive_pre_prepare(msg)
                    case MessageType.PREPARE:
                        self.receive_prepare(msg)
                    case MessageType.COMMIT:
                        self.receive_commit(msg)
                    case MessageType.VIEW_CHANGE:
                        if (self.is_primary()):
                            self.receive_view_change(data)
                    case MessageType.OPERATIONS_DICTIONARY:
                        self.receive_operations_to_commit(data)
                    case MessageType.NEW_VIEW:
                        self.receive_new_view(data)
                    case _:
                        raise Exception("Invalid message type")

    def receive_request(self, data : RequestMessage):
        """
        Process a client request message. If the request is valid, the replica multicasts a pre-prepare message to the other replicas. If the request is a duplicate, the replica sends the cached reply to the client. If the request is out of order, the replica ignores the request.

        Args:
            data (RequestMessage): The client request message.
        """
        client_a = self.find_client(data.get_client_id())
        if (self.reply_cache.get_reply(data.get_client_id()) == None or data.to_string() != self.reply_cache.get_reply(data.get_client_id()).to_string()):
            if (self.hcd.get_chain() != [] and data.get_timestamp() < self.hcd.get_last_block().get_timestamp()):
                return
            elif (self.hcd.get_chain() != [] and data.get_timestamp() == self.hcd.get_last_block().get_timestamp()):
                self.send_reply(data.get_client_id(), self.reply_cache.get_reply(data.get_client_id()))
            else:
                if (client_a.get_version_vector().get_current_system_state(self.f) == None or client_a.get_version_vector().get_current_system_state(self.f).get_hcd() == self.reply_cache.get_reply(data.get_client_id()).get_entry().get_hcd()):
                    self.accepted_requests.append(data)
                    if (self.is_primary()):
                        if (self.hcd.get_chain() == []):
                            sequence_number = 0
                        else:
                            sequence_number = self.hcd.get_last_block().get_sequence_number() + 1
                        pre_prepare_message = self.create_message(MessageType.PRE_PREPARE, view=self.view, sequence_number=sequence_number, message=data)
                        self.multicast_message(pre_prepare_message)
                else:
                    return
        else:
            return
        self.start_timer()

    def receive_pre_prepare(self, data : PrePrepareMessage):
        """
        Process a pre-prepare message. If the message is valid, the replica multicasts a prepare message to the other replicas.

        Args:
            data (PrePrepareMessage): The pre-prepare message."""
        replica = self.find_replica(data.get_primary_replica_id())
        if (replica not in self.replicas):
            self.add_replica(replica)
        if self.received_pre_prepare_messages == []:
            self.received_pre_prepare_messages.append(data)
        for received_pre_prepare in self.received_pre_prepare_messages:
            if received_pre_prepare.get_sequence_number() == data.get_sequence_number():
                if received_pre_prepare.get_request_msg() != data.get_request_msg():
                    return
        if len(self.hcd.get_chain()) > 0:
            if data.get_sequence_number() > self.hcd.get_last_block().get_sequence_number() + 2 or data.get_sequence_number() < self.hcd.get_last_block().get_sequence_number() - 2:
                return
        self.received_pre_prepare_messages.append(data)
        prepare_message = self.create_message(MessageType.PREPARE, view=self.view, sequence_number=data.get_sequence_number(), message=data.get_request_msg())
        self.multicast_message(prepare_message)

    def receive_prepare(self, data: PrepareMessage):
        """
        Process a prepare message. If the replica has received 2f matching prepare messages, it multicasts a commit message to the other replicas.

        Args:
            data (PrepareMessage): The prepare message.
        """
        total_matching_prepare = 0
        total_matching_sequence_number = 0
        replica = self.find_replica(data.get_replica_id())
        if (replica not in self.replicas):
            self.add_replica(replica)
        if data not in self.received_prepare_messages:
            self.received_prepare_messages.append(data)
        for received_prepare in self.received_prepare_messages:
            if received_prepare.get_request_msg() == data.get_request_msg():
                total_matching_prepare += 1
        if total_matching_prepare >= (self.f * 2):
            for received_pre_prepare in self.received_pre_prepare_messages:
                if received_pre_prepare.get_sequence_number() == data.get_sequence_number():
                    total_matching_sequence_number += 1
        else:
            return
        if total_matching_sequence_number >= (self.f * 2):
            self.hcd.add_block(data.get_request_msg(), data.get_sequence_number())
            self.version_vector.update_entry(self.replica_id, self.replica_id, self.view, data.get_sequence_number(), self.hcd)
            commit_message = self.create_message(MessageType.COMMIT, view=self.view, sequence_number=data.get_sequence_number(), hcd=self.hcd)
            self.multicast_message(commit_message)
        else:
            return

    def receive_commit(self, data: CommitMessage):
        """
        Process a commit message. If the replica has received 2f + 1 matching commit messages, it executes the request operation and multicasts a reply message to the client.

        Args:
            data (CommitMessage): The commit message.
        """
        replica = self.find_replica(data.get_replica_id())
        if replica not in self.replicas:
            self.add_replica(replica)
        total = 0
        self.version_vector.update_entry(self.replica_id, data.get_replica_id(), self.view, data.get_sequence_number(), data.get_hcd())
        if data not in self.received_commit_messages:
            self.received_commit_messages.append(data)
        for received_commit in self.received_commit_messages:
            if received_commit.get_sequence_number() == data.get_sequence_number() and received_commit.get_hcd() == data.get_hcd():
                total += 1
        if total >= ((self.f * 2) + 1):
            if len(self.accepted_requests) > 0:
                msg_n = self.accepted_requests[len(self.accepted_requests)-1]
            else:
                msg_n = self.accepted_requests[0]
            operation = msg_n.get_operation()
            client_id = msg_n.get_client_id()
            timestamp = msg_n.get_timestamp()
            result = self.execute_operation(operation)
            if (result):
                reply_message = self.create_message(MessageType.REPLY, client_id=client_id, timestamp=timestamp, result=result, view = self.view, sequence_number = data.get_sequence_number(), message=self.hcd)
                self.update_reply_cache(client_id, reply_message.to_string())
                self.send_reply(client_id, reply_message)
            if data.get_sequence_number() % CHECKPOINT_INTERVAL == 0:
                rcache_n = self.reply_cache.get_reply(client_id)
                E = []
                for replica in self.replicas:
                    if (replica.get_version_vector().is_empty() and replica.get_version_vector().get_entry(self.replica_id, replica.get_replica_id()).get_sequence_number() <= data.get_sequence_number() - (2*I)):
                        E.append(self.sign(replica.get_version_vector().get_entry(self.replica_id, replica.get_replica_id())))
                checkpoint_message = CheckpointMessage(self.replica_id, data.get_sequence_number(), rcache_n, self.version_vector, E)
                self.pending_checkpoints[data.get_sequence_number()] = checkpoint_message
                self.multicast_message(checkpoint_message)

    def receive_view_change(self, data: ViewChangeMessage):
        """
        Process a view change message. If the replica has received 2f + 1 matching view change messages, it multicasts a new view message to the other replicas.

        Args:
            data (ViewChangeMessage): The view change message.
        """
        self.received_view_change_messages.append(data)
        if (data.get_version_vector_entry().get_view() < self.get_version_vector().get_entry(self.get_replica_id(), self.get_replica_id()).get_view() or data.get_version_vector_entry().get_sequence_number() < self.get_version_vector().get_entry(self.get_replica_id, self.get_replica_id()).get_sequence_number()):
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(data.get_version_vector_entry().get_replica().get_host(), data.get_version_vector_entry().get_replica().get_port())
                s.send((MessageType.PRIMARY_NOT_DOMINANT).encode())

        past_view_messages = [msg for msg in self.received_view_change_messages if msg.get_view() == data.get_view()]
        non_conflicting_view_change_messages = [] # V in the paper
        for past_message in past_view_messages:
            for past_message_1 in past_view_messages:
                if past_message.get_sequence_number() == past_message_1.get_sequence_number():
                    if past_message.get_P() != past_message_1.get_P():
                        return
                    else:
                        if past_message not in non_conflicting_view_change_messages:
                            non_conflicting_view_change_messages.append(past_message)
                        if past_message_1 not in non_conflicting_view_change_messages:
                            non_conflicting_view_change_messages.append(past_message_1)
        O = []
        min_s = non_conflicting_view_change_messages[0].get_version_vector_entry().get_sequence_number()
        for non_conflicting_msg in non_conflicting_view_change_messages:
            if non_conflicting_msg.get_version_vector_entry().get_sequence_number() < min_s:
                min_s = non_conflicting_msg.get_version_vector_entry().get_sequence_number()
        max_s = -1
        for non_conflicting_msg in non_conflicting_view_change_messages:
            for P_m in non_conflicting_msg.get_P():
                for msg in P_m:
                    if msg.get_sequence_number() > max_s:
                        max_s = msg.get_sequence_number()
        for n in range(min_s, max_s):
            for non_conflicting_msg in non_conflicting_view_change_messages:
                for P_m in non_conflicting_msg.get_P():
                    if P_m[0].get_sequence_number() == n:
                        pre_prepare_message = self.create_message(MessageType.PRE_PREPARE, data.get_view(), n, P_m[0].get_hashed_request_msg())
                    else:
                        pre_prepare_message = self.create_message(MessageType.PRE_PREPARE, data.get_view(), n, None)
                    O.append(pre_prepare_message)
        new_view_message = NewViewMessage(data.get_view(), non_conflicting_view_change_messages, O)
        self.multicast_message(new_view_message)

    def receive_operations_to_commit(self, data):
        dict = data.get_dict()
        for n in dict:
            self.receive_commit(dict[n])

    def receive_new_view(self, data: NewViewMessage):
        """
        Process a new view message. If the replica has received 2f + 1 matching new view messages, it updates its state and multicasts the pre-prepare messages to the other replicas.

        Args:
            data (NewViewMessage): The new view message.
        """
        if data.get_view() <= self.view:
            return
        self.view = data.get_view()
        for non_conflicting_view_change_message in data.get_V():
            if (isinstance(data.get_O(), list) and data.get_O()[0].get_message_type() == MessageType.PRE_PREPARE and data.get_O()[1:].get_message_type() == MessageType.PREPARE):
                for msg in data.get_O():
                    if msg != data.get_O()[0]:
                        self.multicast_message(msg)

    def execute_operation(self, operation):
        """
        Execute the operation and return the result.

        Returns: bool: True if the operation was successful, False otherwise.
        """
        executed = False
        print(f"Replica {self.replica_id} executed operation: {operation}")
        executed = True
        return executed

    def send_reply(self, client_id, reply_message):
        """
        Send a reply message to the client that issued the request.

        Args:
            client_id (int): The ID of the client.
            reply_message (ReplyMessage): The reply message.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect((self.find_client(client_id).get_host(), self.find_client(client_id).get_port()))
            s.send(reply_message.encode())

    def find_client(self, client_id: int) -> Client:
        """
        Find a client by ID.

        Args:
            client_id (int): The ID of the client.

        Returns:
            Client: The client with the given ID."""
        for client in self.clients:
            if client.get_client_id() == client_id:
                return client
        return 'Client not found'

    def find_replica(self, replica_id: int) -> 'Replica':
        """
        Find a replica by ID.

        Args:
            replica_id (int): The ID of the replica.

        Returns:
            Replica: The replica with the given ID."""
        for replica in self.replicas:
            if replica.get_replica_id() == replica_id:
                return replica
        return 'Replica not found'

    def find_primary_replica(self):
        """
        Find the primary replica.

        Returns:
            Replica: The primary replica.
        """
        for replica in self.replicas:
            if replica.is_primary():
                return replica

    def create_message(self, message_type, view: int = None, sequence_number: int = None, message = None, client_id: int = None, timestamp: float = None, result = None, hcd: HashChainDigest = None):
        """
        Create a message based on the message type.

        Args:
            message_type (MessageType): The type of the message.
            view (int): The view number.
            sequence_number (int): The sequence number assigned by the primary replica.
            message (str): The message content.
            client_id (int): The ID of the client.
            timestamp (float): The timestamp of the request.
            result (bool): The result of the operation.
            hcd (HashChainDigest): The hash chain digest after n operations.

        Returns:
            Message: The message object.

        Throws:
            Exception: If the message type is invalid.
        """
        match message_type:
            case MessageType.PRE_PREPARE:
                return PrePrepareMessage(self.replica_id, view, sequence_number, message)
            case MessageType.PREPARE:
                return PrepareMessage(self.replica_id, view, sequence_number, message)
            case MessageType.COMMIT:
                return CommitMessage(self.replica_id, view, sequence_number, hcd)
            case MessageType.REPLY:
                return ReplyMessage(client_id, timestamp, result, self.replica_id, view, sequence_number, message)
            case _:
                raise Exception("Invalid message type")

    def multicast_message(self, message):
        """
        Multicast a message to all replicas in the system.

        Args:
            message (Message): The message to be multicast.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            for replica in self.replicas:
                s.connect((replica.get_host(), replica.get_port()))
                s.send(message.encode())

    def send_view_change(self):
        """
        Send a view change message to the other replicas.
        """
        if not self.version_vector.is_empty():
            version_vector_r_r = self.version_vector.get_entry(self.replica_id, self.replica_id) #  version vector entry for r’s last committed operation
            new_view = self.view + 1 # new view
            P = [] # set of sets Pm.  Each Pm contains the pre-prepare message with sequence number > m and 2f corresponding matching prepare messages.
            for preprepare_message in self.received_pre_prepare_messages:
                list = []
                if preprepare_message.get_sequence_number() > self.hcd.get_last_block().get_sequence_number() and len(self.received_prepare_messages) > 2*self.f:
                    list.append(preprepare_message)
                    for prepare_message in self.received_prepare_messages:
                        if prepare_message.get_sequence_number() == preprepare_message.get_sequence_number():
                            list.append(self.received_prepare_messages)
                    P.append(list)
            vv_replica_id = version_vector_r_r.get_replica_id()
            vv_view = version_vector_r_r.get_view()
            vv_sequence_number = version_vector_r_r.get_sequence_number()
            vv_hcd = version_vector_r_r.get_hcd()
            view_change_message = ViewChangeMessage(new_view, self.replica_id, vv_replica_id, vv_view, vv_sequence_number, vv_hcd, P)
            self.multicast_message(view_change_message)
            self.start_timer()

    def deserialize_message(self, msg):
        """
        Deserialize a message string and return the corresponding message object. Messages are serialized as strings upon receiving, thus they need to be deserialized before processing. Each is deserialized based on the message type.

        Args:
            msg (str): The message string.

        Returns:
            Message: The message object.

        Throws:
            Exception: If the message is invalid.
        """
        try:
            components = msg.split(', ')
            if components[0].split(': ')[1] == 'MessageType.REQUEST':
                return RequestMessage(components[4].split(': ')[1], float(components[5].split(': ')[1]), int(components[3].split(': ')[1]), components[6].split(': ')[1])
            elif components[0].split(': ')[1] == 'MessageType.PRE_PREPARE':
                return PrePrepareMessage(int(components[3].split(': ')[1]), int(components[1].split(': ')[1]), int(components[2].split(': ')[1]), components[4].split(': ')[1])
            elif components[0].split(': ')[1] == 'MessageType.PREPARE':
                return PrepareMessage(int(components[3].split(': ')[1]), int(components[1].split(': ')[1]), int(components[2].split(': ')[1]), components[4].split(': ')[1])
            elif components[0].split(': ')[1] == 'MessageType.COMMIT':
                return CommitMessage(int(components[3].split(': ')[1]), int(components[1].split(': ')[1]), int(components[2].split(': ')[1]), components[4].split(': ')[1])
            elif components[0].split(': ')[1] == 'MessageType.VIEW_CHANGE':
                return ViewChangeMessage(int(components[1].split(': ')[1]), int(components[2].split(': ')[1]), int(components[3].split(': ')[1]), int(components[4].split(': ')[1]), int(components[5].split(': ')[1]), components[6].split(': ')[1], list(components[7].split(': ')[1]))
            elif components[0].split(': ')[1] == 'MessageType.NEW_VIEW':
                return NewViewMessage(int(components[1].split(': ')), components[2].split(': '), components[3].split(': '))
            else:
                raise Exception("Invalid message type")
        except Exception as e:
            return e
