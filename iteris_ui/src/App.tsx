import { lazy, Suspense } from 'react';
import { Navigate, Route, Routes } from 'react-router-dom';
import { ROUTES } from './routes';

/**
 * Top-level router. Each page is code-split so the heavy Three.js landing bundle
 * never loads on the clinical pages. Pages are built by dedicated agents under
 * src/pages/* and all consume the shared tokens + API contract.
 */
const Landing = lazy(() => import('./pages/Landing'));
const Research = lazy(() => import('./pages/Research'));
const Workspace = lazy(() => import('./pages/Workspace'));
const ModelLibrary = lazy(() => import('./pages/ModelLibrary'));
const DatasetExplorer = lazy(() => import('./pages/DatasetExplorer'));

export default function App() {
  return (
    <Suspense fallback={<div className="p-8 font-body text-muted">Loading…</div>}>
      <Routes>
        <Route path={ROUTES.landing} element={<Landing />} />
        <Route path={ROUTES.research} element={<Research />} />
        <Route path={ROUTES.workspace} element={<Workspace />} />
        <Route path={ROUTES.models} element={<ModelLibrary />} />
        <Route path={ROUTES.datasets} element={<DatasetExplorer />} />
        <Route path="*" element={<Navigate to={ROUTES.landing} replace />} />
      </Routes>
    </Suspense>
  );
}
