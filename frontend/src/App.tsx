import { BrowserRouter, Routes, Route, Navigate, NavLink } from 'react-router-dom';
import AdminPage from './pages/AdminPage';
import AnalystPage from './pages/AnalystPage';

function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-gray-50 flex flex-col">
        <nav className="bg-gray-900 text-white px-6 py-3 flex items-center gap-6">
          <h1 className="text-lg font-bold tracking-tight">AST PoC</h1>
          <div className="flex gap-1">
            <NavLink
              to="/admin"
              className={({ isActive }) =>
                `px-3 py-1.5 rounded text-sm ${isActive ? 'bg-gray-700 text-white' : 'text-gray-400 hover:text-white'}`
              }
            >
              Admin
            </NavLink>
            <NavLink
              to="/analyst"
              className={({ isActive }) =>
                `px-3 py-1.5 rounded text-sm ${isActive ? 'bg-gray-700 text-white' : 'text-gray-400 hover:text-white'}`
              }
            >
              Analyst
            </NavLink>
          </div>
        </nav>

        <main className="flex-1">
          <Routes>
            <Route path="/admin" element={<AdminPage />} />
            <Route path="/analyst" element={<AnalystPage />} />
            <Route path="*" element={<Navigate to="/analyst" replace />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}

export default App;
