#! /usr/bin/env python
import pylab as plt
import numpy as np
import argparse
import queue
import socket
from threading import Thread


def plot(queu: queue.Queue):
    """Plot data from a queue"""
    # Enable Matplotlib interactive mode
    plt.ion()
    axes = plt.figure().add_subplot(111)
    graph = axes.plot([], [])[0]
    while True:
        # Plot 8 * 64 time steps at a time
        buf = np.concatenate([queu.get() for _ in range(8)])
        # Update the X and Y data
        graph.set_data(buf.reshape(-1, 2).T)
        # Rescale the plotting area to match the data
        axes.relim()
        axes.autoscale_view(True, True, True)
        plt.draw()
        # Pause to do not flood the display with refresh
        plt.pause(0.1)


# https://stackoverflow.com/questions/17667903/python-socket-receive-large-amount-of-data
def recvall(sock, n) -> bytearray:
    """Helper function to recv n bytes or return None if EOF is hit"""
    data = bytearray()
    while len(data) < n:
        data.extend(sock.recv(n - len(data)))
    return data


def receive_and_queue(sock: socket.socket, output_file, queu):
    """Receive data from a socket, save them to a file and push them to a queue"""
    while True:
        # Read 64 time steps at a time
        d = recvall(sock, 8 * 2 * 64)
        data = np.frombuffer(d)
        if output_file:
            data.tofile(output_file)
        try:
            queu.put(data, block=False)
        except queue.Full:
            # The main thread (i.e Matplotlib) is not popping the queue fast enough.
            # Not an issue, we just drop some time steps.
            pass


def client(args):
    """Connect to a server"""
    if args.outputfile:
        ofile = open(args.outputfile, "wb")
    else:
        ofile = None
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # A queue to send data from the receiving thread to Matplotlib (which must
        # be in the main thread)
        queu = queue.Queue()
        s.connect((args.host, args.port))
        print("Connected to server")
        thread = Thread(target=receive_and_queue, args=[s, ofile, queu])
        thread.start()
        plot(queu)


def serve_sin_wave(
    conn,
    addr,
    frequency: float = 1.0,
    amplitude: float = 1.0,
    sample_rate: int = 100,
):
    """Generate a sinusoidal signal and send it to a client"""
    t = np.linspace(0, 1.0, sample_rate, endpoint=False)
    sin_wave = amplitude * np.sin(2 * np.pi * frequency * t)
    period = 0
    try:
        while True:
            buff = np.vstack([t + period, sin_wave]).T.reshape(-1)
            conn.sendall(buff)
            period += 1
    except (ConnectionResetError, BrokenPipeError):
        print(f"Connection with {addr} lost.")
    finally:
        conn.close()


def server(args):
    """Wait for connection of a plotter"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((args.host, args.port))
        s.listen()
        print(f"Server listening on {args.host}:{args.port}")
        while True:
            conn, addr = s.accept()
            print(f"Accepted connection from {addr}")
            client_thread = Thread(target=serve_sin_wave, args=(conn, addr))
            client_thread.start()


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    plotter_cmd = subparsers.add_parser("plotter", help="Run the plotter")
    plotter_cmd.add_argument(
        "--host", type=str, default="localhost", help="Host address"
    )
    plotter_cmd.add_argument("--port", type=int, default=8000, help="Port number")
    plotter_cmd.add_argument("--outputfile", type=str, help="Path to the output file")
    plotter_cmd.set_defaults(func=client)

    server_cmd = subparsers.add_parser("server", help="Run the demo server")
    server_cmd.add_argument(
        "--host", type=str, default="localhost", help="Host address"
    )
    server_cmd.add_argument("--port", type=int, default=8000, help="Port number")
    server_cmd.set_defaults(func=server)

    args = parser.parse_args()
    if args.command:
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
