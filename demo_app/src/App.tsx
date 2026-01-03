import React, { useEffect, useRef } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import "./styles.css";
import AppShell from "./app/AppShell";
import { connectWSRobust } from "./api/ws";
import { getWsUrl, useBackendHealthQuery } from "./api/client";
import { useDemoStore } from "./state/useDemoStore";
import type { InferenceTick, StatusMessage } from "./types/schema";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1
    }
  }
});

function AppInner() {
  const setBackendReachable = useDemoStore((state) => state.setBackendReachable);
  const setWsState = useDemoStore((state) => state.setWsState);
  const setLastWsMessageAt = useDemoStore((state) => state.setLastWsMessageAt);
  const setWs = useDemoStore((state) => state.setWs);
  const setTick = useDemoStore((state) => state.setTick);
  const setStatus = useDemoStore((state) => state.setStatus);
  const appendTimeline = useDemoStore((state) => state.appendTimeline);
  const pushEvent = useDemoStore((state) => state.pushEvent);

  const healthQuery = useBackendHealthQuery();

  useEffect(() => {
    if (healthQuery.isFetching) {
      setBackendReachable(null);
    } else if (healthQuery.isError) {
      setBackendReachable(false);
    } else if (healthQuery.isSuccess) {
      setBackendReachable(true);
    }
  }, [healthQuery.isFetching, healthQuery.isError, healthQuery.isSuccess, setBackendReachable]);

  const wsGuardRef = useRef(false);
  const lastTickAtRef = useRef(0);
  const pendingTickRef = useRef<InferenceTick | null>(null);
  const throttleTimerRef = useRef<number | null>(null);

  useEffect(() => {
    if (wsGuardRef.current) return;
    wsGuardRef.current = true;

    const wsUrl = getWsUrl();
    setWs(null);

    const applyTick = (tick: InferenceTick) => {
      setTick(tick);
      appendTimeline(tick.prediction.action_name);
    };

    const flushPending = () => {
      if (!pendingTickRef.current) return;
      applyTick(pendingTickRef.current);
      pendingTickRef.current = null;
      lastTickAtRef.current = Date.now();
    };

    const connection = connectWSRobust(
      wsUrl,
      {
        onTick: (tick) => {
          const now = Date.now();
          const elapsed = now - lastTickAtRef.current;
          setLastWsMessageAt(now);

          if (elapsed >= 50) {
            lastTickAtRef.current = now;
            applyTick(tick);
            return;
          }

          pendingTickRef.current = tick;
          if (throttleTimerRef.current === null) {
            throttleTimerRef.current = window.setTimeout(() => {
              throttleTimerRef.current = null;
              flushPending();
            }, Math.max(0, 50 - elapsed));
          }
        },
        onStatus: (status: StatusMessage) => {
          setStatus(status);
          setLastWsMessageAt(Date.now());
          pushEvent({ level: status.level, message: status.message });
        },
        onStateChange: (state) => {
          setWsState(state);
        },
        onOpen: () => {
          setWs(connection.getSocket());
          pushEvent({ level: "info", message: "WebSocket connected" });
        },
        onClose: () => {
          setWs(null);
          pushEvent({ level: "warning", message: "WebSocket closed" });
        },
        onError: () => {
          pushEvent({ level: "error", message: "WebSocket error" });
        }
      },
      {
        initialBackoffMs: 500,
        maxBackoffMs: 8000,
        jitterMs: 400
      }
    );

    return () => {
      connection.close();
      if (throttleTimerRef.current !== null) {
        window.clearTimeout(throttleTimerRef.current);
        throttleTimerRef.current = null;
      }
      wsGuardRef.current = false;
    };
  }, [appendTimeline, pushEvent, setLastWsMessageAt, setStatus, setTick, setWs, setWsState]);

  return <AppShell />;
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AppInner />
    </QueryClientProvider>
  );
}
