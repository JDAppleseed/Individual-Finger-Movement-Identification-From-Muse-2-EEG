import React, { Suspense } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Cuboid, Network } from "lucide-react";
import clsx from "clsx";
import { useDemoStore } from "../../state/useDemoStore";

const NnStage = React.lazy(() => import("../viz2d/NnStage"));
const HandStage = React.lazy(() => import("../viz3d/HandStage"));

const stageVariants = {
  initial: { opacity: 0, y: 12 },
  animate: { opacity: 1, y: 0, transition: { duration: 0.35 } },
  exit: { opacity: 0, y: -8, transition: { duration: 0.2 } }
};

export default function CenterStage() {
  const stage = useDemoStore((state) => state.stage);
  const setStage = useDemoStore((state) => state.setStage);

  return (
    <section className="center-stage">
      <div className="stage-header">
        <div>
          <h2>Neural Stage</h2>
          <p>Switch between introspection and spatial intent visualization.</p>
        </div>
        <div className="stage-toggle">
          <button
            className={clsx("stage-toggle-btn", stage === "nn" && "active")}
            onClick={() => setStage("nn")}
          >
            <Network size={16} /> 2D Network
          </button>
          <button
            className={clsx("stage-toggle-btn", stage === "hand" && "active")}
            onClick={() => setStage("hand")}
          >
            <Cuboid size={16} /> 3D Hand
          </button>
        </div>
      </div>

      <div className="stage-body">
        <Suspense fallback={<div className="stage-loading">Loading stage...</div>}>
          <AnimatePresence mode="wait">
            {stage === "nn" ? (
              <motion.div key="nn" className="stage-panel" variants={stageVariants} initial="initial" animate="animate" exit="exit">
                <NnStage />
              </motion.div>
            ) : (
              <motion.div key="hand" className="stage-panel" variants={stageVariants} initial="initial" animate="animate" exit="exit">
                <HandStage />
              </motion.div>
            )}
          </AnimatePresence>
        </Suspense>
      </div>
    </section>
  );
}
