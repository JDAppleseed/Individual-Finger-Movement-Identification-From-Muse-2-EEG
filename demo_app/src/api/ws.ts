import type { InferenceTick, StatusMessage } from "../types/schema";
import { parseMessage } from "../types/schema";

export type WSHandlers = {
  onTick: (tick: InferenceTick) => void;
  onStatus: (status: StatusMessage) => void;
  onRaw?: (payload: unknown) => void;
};

export function connectWS(url: string, handlers: WSHandlers) {
  const ws = new WebSocket(url);

  ws.onmessage = (event) => {
    let payload: unknown;
    try {
      payload = JSON.parse(event.data as string);
    } catch (err) {
      return;
    }
    handlers.onRaw?.(payload);
    const parsed = parseMessage(payload);
    if (!parsed) {
      return;
    }
    if (parsed.type === "tick") {
      handlers.onTick(parsed);
    } else {
      handlers.onStatus(parsed);
    }
  };

  return ws;
}
