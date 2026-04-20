import { NavLink } from "react-router-dom";
import { MessageSquare, BarChart3, ScrollText, Info } from "lucide-react";

const links = [
  { to: "/", icon: MessageSquare, label: "Chat" },
  { to: "/evaluate", icon: BarChart3, label: "Evaluation" },
  { to: "/logs", icon: ScrollText, label: "Logs" },
  { to: "/about", icon: Info, label: "About" },
];

export default function Sidebar() {
  return (
    <aside className="w-64 h-screen flex flex-col glass border-r border-fintech-600/20 fixed left-0 top-0 z-10">
      {/* Logo */}
      <div className="p-6 border-b border-fintech-600/20">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-fintech-accent to-fintech-400 flex items-center justify-center">
            <span className="text-white font-bold text-sm">Q</span>
          </div>
          <div>
            <h1 className="text-white font-bold text-sm tracking-wide">QFIS</h1>
            <p className="text-fintech-200/50 text-xs">Financial AI System</p>
          </div>
        </div>
      </div>

      {/* Nav */}
      <nav className="flex-1 p-4 space-y-1">
        {links.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === "/"}
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm transition-all duration-200
              ${isActive
                ? "bg-fintech-400/20 text-fintech-accent border border-fintech-400/30"
                : "text-slate-400 hover:text-slate-200 hover:bg-fintech-700/50"
              }`
            }
          >
            <Icon size={16} />
            {label}
          </NavLink>
        ))}
      </nav>

      {/* Footer */}
      <div className="p-4 border-t border-fintech-600/20">
        <div className="glass rounded-xl p-3">
          <p className="text-xs text-fintech-200/50 font-mono">Phi-2 + QLoRA + RAG</p>
          <p className="text-xs text-fintech-accent/70 mt-0.5">FinQA Dataset</p>
        </div>
      </div>
    </aside>
  );
}
