import { BrowserRouter, Routes, Route } from "react-router-dom";
import Sidebar from "./components/Sidebar";
import Chat from "./pages/Chat";
import Evaluate from "./pages/Evaluate";
import Logs from "./pages/Logs";
import About from "./pages/About";
import "./index.css";

export default function App() {
  return (
    <BrowserRouter>
      <div className="flex bg-fintech-900 min-h-screen">
        <Sidebar />
        <main className="flex-1 ml-64">
          <Routes>
            <Route path="/" element={<Chat />} />
            <Route path="/evaluate" element={<Evaluate />} />
            <Route path="/logs" element={<Logs />} />
            <Route path="/about" element={<About />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}
