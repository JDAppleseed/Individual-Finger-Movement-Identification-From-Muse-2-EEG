import type { InferenceTick, StatusMessage } from "../types/schema";
import { parseMessage } from "../types/schema";

export type WsState = "connecting" | "open" | "closed" | "error";

export type WSHandlers = {
  onTick: (tick: InferenceTick) => void;
  onStatus: (status: StatusMessage) => void;
  onRaw?: (payload: unknown) => void;
  onStateChange?: (state: WsState, event?: Event | CloseEvent) => void;
  onOpen?: (event: Event, socket: WebSocket) => void;
  onClose?: (event: CloseEvent) => void;
  onError?: (event: Event) => void;
};

export type WSOptions = {
  reconnect?: boolean;
  initialBackoffMs?: number;
  maxBackoffMs?: number;
  jitterMs?: number;
  maxRetries?: number;
};

export type WSConnection = {
  close: () => void;
  getSocket: () => WebSocket | null;
  getState: () => WsState;
};

export function connectWSRobust(url: string, handlers: WSHandlers, options: WSOptions = {}): WSConnection {
  const opts = {
    reconnect: options.reconnect ?? true,
    initialBackoffMs: options.initialBackoffMs ?? 400,
    maxBackoffMs: options.maxBackoffMs ?? 8000,
    jitterMs: options.jitterMs ?? 250,
    maxRetries: options.maxRetries ?? Number.POSITIVE_INFINITY
  };

  let socket: WebSocket | null = null;
  let state: WsState = "connecting";
  let retryCount = 0;
  let closedByUser = false;
  let reconnectTimer: number | null = null;

  const updateState = (next: WsState, event?: Event | CloseEvent) => {
    state = next;
    handlers.onStateChange?.(next, event);
  };

  const scheduleReconnect = () => {
    if (!opts.reconnect || closedByUser) return;
    if (retryCount >= opts.maxRetries) return;
    const backoff = Math.min(opts.maxBackoffMs, opts.initialBackoffMs * 2 ** retryCount);
    const jitter = Math.random() * opts.jitterMs;
    reconnectTimer = window.setTimeout(() => {
      retryCount += 1;
      connect();
    }, backoff + jitter);
  };

  const connect = () => {
    if (closedByUser) return;
    updateState("connecting");
    socket = new WebSocket(url);

    socket.onopen = (event) => {
      retryCount = 0;
      updateState("open", event);
      handlers.onOpen?.(event, socket!);
    };

    socket.onmessage = (event) => {
      let payload: unknown;
      try {
        payload = JSON.parse(event.data as string);
      } catch {
        return;
      }
      handlers.onRaw?.(payload);
      const parsed = parseMessage(payload);
      if (!parsed) return;
      if (parsed.type === "tick") {
        handlers.onTick(parsed);
      } else {
        handlers.onStatus(parsed);
      }
    };

    socket.onerror = (event) => {
      updateState("error", event);
      handlers.onError?.(event);
    };

    socket.onclose = (event) => {
      updateState("closed", event);
      handlers.onClose?.(event);
      socket = null;
      scheduleReconnect();
    };
  };

  connect();

  return {
    close: () => {
      closedByUser = true;
      if (reconnectTimer !== null) {
        window.clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }
      if (socket && socket.readyState !== WebSocket.CLOSED) {
        socket.close();
      }
      socket = null;
      updateState("closed");
    },
    getSocket: () => socket,
    getState: () => state
  };
}
