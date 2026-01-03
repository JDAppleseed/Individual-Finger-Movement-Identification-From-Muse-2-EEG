import React, { Suspense } from "react";
import { shallow } from "zustand/shallow";
import { useDemoStore } from "../../state/useDemoStore";

const NnVisualizer = React.lazy(() => import("../../pages/NnVisualizer"));

export default function NnStage() {
  const { ws, tick, selectedNodeId, setSelectedNode, setHoveredNode } = useDemoStore(
    (state) => ({
      ws: state.ws,
      tick: state.tick,
      selectedNodeId: state.selectedNodeId,
      setSelectedNode: state.setSelectedNode,
      setHoveredNode: state.setHoveredNode
    }),
    shallow
  );

  return (
    <div className="nn-stage">
      <div className="stage-scroll">
        <Suspense fallback={<div className="stage-loading">Loading NN Visualizer...</div>}>
          <NnVisualizer
            ws={ws}
            tick={tick}
            selectedNodeId={selectedNodeId}
            onSelectNode={setSelectedNode}
            onHoverNode={setHoveredNode}
          />
        </Suspense>
      </div>
    </div>
  );
}
