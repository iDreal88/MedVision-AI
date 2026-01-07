import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Upload,
  Brain,
  ChevronRight,
  FileText,
  Download,
  Image as ImageIcon,
  Zap,
  AlertCircle,
  Activity,
  CheckCircle2,
  X
} from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

function App() {
  const [selectedModel, setSelectedModel] = useState('CNN+CLAHE');
  const [models, setModels] = useState([]);
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [currentView, setCurrentView] = useState('home'); // 'home', 'dashboard', 'documentation', 'log'
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [analysisHistory, setAnalysisHistory] = useState(() => {
    const saved = localStorage.getItem('analysisHistory');
    return saved ? JSON.parse(saved) : [];
  });

  const renderFormattedText = (text) => {
    return text.split(/(\*\*.*?\*\*)/g).map((part, i) => {
      if (part.startsWith('**') && part.endsWith('**')) {
        return <strong key={i} className="font-bold text-white/90">{part.slice(2, -2)}</strong>;
      }
      return part;
    });
  };

  useEffect(() => {
    localStorage.setItem('analysisHistory', JSON.stringify(analysisHistory));
  }, [analysisHistory]);

  useEffect(() => {
    fetchModels();
  }, []);

  const fetchModels = async () => {
    try {
      const resp = await axios.get(`${API_BASE}/models`);
      setModels(resp.data);
    } catch (err) {
      console.error("Failed to fetch models", err);
      setError(`Backend API not reachable at ${API_BASE}. Please ensure your Vercel VITE_API_URL matches your Railway URL.`);
    }
  };

  const handleFileChange = (e) => {
    const f = e.target.files[0];
    if (f) {
      setFile(f);
      setPreview(URL.createObjectURL(f));
      setResult(null);
    }
  };

  const runDiagnosis = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('model_name', selectedModel);

    try {
      const resp = await axios.post(`${API_BASE}/predict`, formData);
      setResult(resp.data);
      setAnalysisHistory(prev => [{
        id: Date.now(),
        timestamp: new Date().toLocaleTimeString(),
        model: selectedModel,
        label: resp.data.label,
        confidence: resp.data.confidence,
        preview: `data:image/jpeg;base64,${resp.data.processed_image}`
      }, ...prev]);
    } catch (err) {
      setError(err.response?.data?.detail || "An error occurred during diagnosis.");
    } finally {
      setLoading(false);
    }
  };

  const downloadPDF = async () => {
    if (!result?.report) return;
    try {
      const resp = await axios.post(`${API_BASE}/download-pdf`,
        { content: result.report },
        { responseType: 'blob' }
      );
      const url = window.URL.createObjectURL(new Blob([resp.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `diagnosis_report_${result.label}.pdf`);
      document.body.appendChild(link);
      link.click();
    } catch (err) {
      console.error("PDF Download Error:", err);
      let errorMessage = "Unknown error";
      if (err.response?.data instanceof Blob) {
        const text = await err.response.data.text();
        try {
          const parsed = JSON.parse(text);
          errorMessage = parsed.detail || text;
        } catch (e) {
          errorMessage = text;
        }
      } else {
        errorMessage = err.response?.data?.detail || err.message || "Unknown error";
      }
      alert(`Failed to download PDF: ${errorMessage}`);
    }
  };

  return (
    <div className="min-h-screen bg-[#0a0a0c] text-slate-200 selection:bg-brand-primary/30">
      {/* Background Orbs */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-brand-primary/10 rounded-full blur-[120px] animate-pulse" />
        <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-brand-secondary/10 rounded-full blur-[120px] animate-pulse" style={{ animationDelay: '2s' }} />
      </div>

      <nav className="sticky top-0 z-50 glass border-b border-white/5 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-gradient-to-br from-brand-primary to-brand-secondary rounded-xl flex items-center justify-center shadow-lg shadow-brand-primary/20">
            <Activity className="text-white w-6 h-6" />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight text-white">MedVision AI</h1>
            <p className="text-[10px] text-slate-500 uppercase tracking-[0.2em] font-medium leading-none">Predict Cancer</p>
          </div>
        </div>
        <div className="hidden md:flex items-center gap-8 text-sm font-medium text-slate-400">
          <button onClick={() => setCurrentView('home')} className={`hover:text-white transition-colors ${currentView === 'home' ? 'text-brand-primary' : ''}`}>Diagnosis</button>
          <button onClick={() => setCurrentView('dashboard')} className={`hover:text-white transition-colors ${currentView === 'dashboard' ? 'text-brand-primary' : ''}`}>Dashboard</button>
          <button onClick={() => setCurrentView('documentation')} className={`hover:text-white transition-colors ${currentView === 'documentation' ? 'text-brand-primary' : ''}`}>Documentation</button>
          <button onClick={() => setCurrentView('log')} className={`hover:text-white transition-colors ${currentView === 'log' ? 'text-brand-primary' : ''}`}>Analysis Log</button>
        </div>
        <div className="flex items-center gap-4">
          <div className="hidden md:flex px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-[10px] font-bold uppercase tracking-wider items-center gap-1.5">
            <div className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
            Online
          </div>
          {/* Mobile Menu Toggle */}
          <button onClick={() => setMobileMenuOpen(!mobileMenuOpen)} className="md:hidden text-white p-2">
            {mobileMenuOpen ? <X /> : <div className="space-y-1.5"><div className="w-6 h-0.5 bg-white" /><div className="w-6 h-0.5 bg-white" /><div className="w-6 h-0.5 bg-white" /></div>}
          </button>
        </div>
      </nav>

      {/* Mobile Navigation Dropdown */}
      <AnimatePresence>
        {mobileMenuOpen && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="md:hidden bg-black/90 backdrop-blur-xl border-b border-white/10 overflow-hidden"
          >
            <div className="flex flex-col gap-4 p-6 text-sm font-medium text-slate-400">
              <button onClick={() => { setCurrentView('home'); setMobileMenuOpen(false); }} className={`text-left py-2 hover:text-white transition-colors ${currentView === 'home' ? 'text-brand-primary' : ''}`}>Diagnosis</button>
              <button onClick={() => { setCurrentView('dashboard'); setMobileMenuOpen(false); }} className={`text-left py-2 hover:text-white transition-colors ${currentView === 'dashboard' ? 'text-brand-primary' : ''}`}>Dashboard</button>
              <button onClick={() => { setCurrentView('documentation'); setMobileMenuOpen(false); }} className={`text-left py-2 hover:text-white transition-colors ${currentView === 'documentation' ? 'text-brand-primary' : ''}`}>Documentation</button>
              <button onClick={() => { setCurrentView('log'); setMobileMenuOpen(false); }} className={`text-left py-2 hover:text-white transition-colors ${currentView === 'log' ? 'text-brand-primary' : ''}`}>Analysis Log</button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <main className="relative max-w-7xl mx-auto px-6 py-12">
        <AnimatePresence mode="wait">
          {currentView === 'home' && (
            <motion.div
              key="home"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="grid lg:grid-cols-12 gap-12"
            >

              {/* Left Column: Input */}
              <div className="lg:col-span-5 space-y-8">
                <section className="space-y-4">
                  <div className="flex items-center gap-2 mb-2">
                    <Brain className="w-5 h-5 text-brand-primary" />
                    <h2 className="text-lg font-semibold text-white">Model Configuration</h2>
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    {['CNN+CLAHE', 'ResNet50', 'VGG16', 'VGG19'].map((m) => (
                      <button
                        key={m}
                        onClick={() => setSelectedModel(m)}
                        className={`p-4 rounded-2xl border transition-all duration-300 text-left relative overflow-hidden group ${selectedModel === m
                          ? 'border-brand-primary bg-brand-primary/5 text-white shadow-lg shadow-brand-primary/5'
                          : 'border-white/5 bg-white/[0.02] text-slate-400 hover:border-white/20'
                          }`}
                      >
                        <span className="text-sm font-bold block mb-1">{m}</span>
                        <span className="text-[10px] opacity-60">
                          {m === 'CNN+CLAHE' ? 'Optimized Custom CNN' : 'Transfer Learning'}
                        </span>
                        {selectedModel === m && (
                          <motion.div
                            layoutId="active-pill"
                            className="absolute right-3 top-1/2 -translate-y-1/2"
                          >
                            <CheckCircle2 className="w-4 h-4 text-brand-primary" />
                          </motion.div>
                        )}
                      </button>
                    ))}
                  </div>
                </section>

                <section className="space-y-4">
                  <div className="flex items-center gap-2 mb-2">
                    <ImageIcon className="w-5 h-5 text-brand-secondary" />
                    <h2 className="text-lg font-semibold text-white">Mammogram Upload</h2>
                  </div>
                  <label
                    className={`flex flex-col items-center justify-center h-64 border-2 border-dashed rounded-[32px] transition-all duration-500 cursor-pointer group ${preview ? 'border-brand-secondary/50 bg-brand-secondary/5' : 'border-white/10 hover:border-white/20 bg-white/[0.02]'
                      }`}
                  >
                    {preview ? (
                      <div className="relative w-full h-full p-4">
                        <img
                          src={preview}
                          className="w-full h-full object-contain rounded-2xl"
                          alt="Preview"
                        />
                        <button
                          onClick={(e) => { e.preventDefault(); setPreview(null); setFile(null); }}
                          className="absolute top-6 right-6 p-2 bg-black/60 backdrop-blur-md rounded-full text-white hover:bg-black/80 transition-all border border-white/10"
                        >
                          <X className="w-4 h-4" />
                        </button>
                      </div>
                    ) : (
                      <div className="flex flex-col items-center gap-3 text-center px-4">
                        <div className="w-14 h-14 rounded-2xl bg-white/5 flex items-center justify-center group-hover:scale-110 transition-transform duration-500 border border-white/10">
                          <Upload className="w-6 h-6 text-slate-400 group-hover:text-brand-secondary" />
                        </div>
                        <div className="space-y-1.5">
                          <p className="font-semibold text-slate-300">Drop medical image</p>
                          <p className="text-xs text-slate-500">Supports PNG, JPG, or DICOM-exported JPEG</p>
                        </div>
                      </div>
                    )}
                    <input type="file" className="hidden" onChange={handleFileChange} accept="image/*" />
                  </label>
                </section>

                <button
                  onClick={runDiagnosis}
                  disabled={!file || loading}
                  className={`w-full py-4 rounded-2xl font-bold flex items-center justify-center gap-2 transition-all duration-500 overflow-hidden relative group ${!file || loading
                    ? 'bg-white/5 text-slate-500 cursor-not-allowed border border-white/5'
                    : 'bg-gradient-to-r from-brand-primary to-brand-secondary text-white shadow-xl shadow-brand-primary/20 hover:scale-[1.02] active:scale-[0.98]'
                    }`}
                >
                  {loading ? (
                    <>
                      <div className="w-5 h-5 border-3 border-white/30 border-t-white rounded-full animate-spin" />
                      <span>Processing Neural Layers...</span>
                    </>
                  ) : (
                    <>
                      <Zap className="w-5 h-5 fill-current" />
                      <span>Execute Neural Diagnosis</span>
                    </>
                  )}
                </button>

                {error && (
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="p-4 rounded-2xl bg-red-500/10 border border-red-500/20 text-red-400 text-sm flex gap-3"
                  >
                    <AlertCircle className="w-5 h-5 shrink-0" />
                    <p>{error}</p>
                  </motion.div>
                )}
              </div>

              {/* Right Column: Output */}
              <div className="lg:col-span-7">
                <AnimatePresence mode="wait">
                  {!result ? (
                    <motion.div
                      key="empty"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      className="h-full min-h-[500px] rounded-[40px] border-2 border-white/5 bg-white/[0.01] flex flex-col items-center justify-center text-center p-8 border-dashed"
                    >
                      <div className="relative mb-6">
                        <Activity className="w-20 h-20 text-white/5" />
                        <motion.div
                          animate={{ scale: [1, 1.2, 1] }}
                          transition={{ duration: 4, repeat: Infinity }}
                          className="absolute inset-0 flex items-center justify-center"
                        >
                          <Zap className="w-8 h-8 text-brand-primary/20" />
                        </motion.div>
                      </div>
                      <h3 className="text-xl font-semibold text-slate-400">Waiting for Data Pipeline</h3>
                      <p className="text-sm text-slate-600 max-w-xs mt-2">Initialize the diagnosis on the left to witness advanced neural analysis results here.</p>
                    </motion.div>
                  ) : (
                    <motion.div
                      key="results"
                      initial={{ opacity: 0, x: 20 }}
                      animate={{ opacity: 1, x: 0 }}
                      className="space-y-8"
                    >
                      {/* Results Header Card */}
                      <div className="grid grid-cols-2 gap-6">
                        <div className="p-8 rounded-[32px] glass hover:border-brand-primary/30 transition-colors group">
                          <p className="text-xs font-bold uppercase tracking-widest text-slate-500 mb-2">Diagnosis Type</p>
                          <div className="flex items-end gap-3">
                            <h3 className={`text-4xl font-black ${result.label === 'Cancer' ? 'text-red-500' : 'text-emerald-400'}`}>
                              {result.label}
                            </h3>
                            <div className={`mb-1 px-3 py-0.5 rounded-full text-[10px] font-bold border ${result.label === 'Cancer' ? 'bg-red-500/10 border-red-500/30 text-red-400' : 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400'
                              }`}>
                              Verified Result
                            </div>
                          </div>
                        </div>
                        <div className="p-8 rounded-[32px] glass hover:border-brand-secondary/30 transition-colors group">
                          <p className="text-xs font-bold uppercase tracking-widest text-slate-500 mb-2">Confidence Level</p>
                          <div className="flex items-end gap-3">
                            <h3 className="text-4xl font-black text-white">
                              {result.confidence.toFixed(1)}%
                            </h3>
                            <div className="mb-1 w-12 h-1.5 rounded-full bg-slate-800 overflow-hidden">
                              <motion.div
                                initial={{ width: 0 }}
                                animate={{ width: `${result.confidence}%` }}
                                transition={{ duration: 1, delay: 0.5 }}
                                className="h-full bg-brand-secondary"
                              />
                            </div>
                          </div>
                        </div>
                      </div>

                      {/* Visualizations */}
                      <div className="grid grid-cols-2 gap-6">
                        <div className="space-y-4">
                          <h4 className="text-sm font-bold text-slate-400 flex items-center gap-2">
                            <Activity className="w-4 h-4 text-emerald-500" />
                            Original Preprocessed
                          </h4>
                          <div className="aspect-square rounded-[32px] overflow-hidden border border-white/5 bg-black shadow-2xl">
                            <img
                              src={`data:image/jpeg;base64,${result.processed_image}`}
                              className="w-full h-full object-cover"
                              alt="Processed"
                            />
                          </div>
                        </div>
                        <div className="space-y-4">
                          <h4 className="text-sm font-bold text-slate-400 flex items-center gap-2">
                            <Zap className="w-4 h-4 text-brand-primary" />
                            Explainable AI (Grad-CAM)
                          </h4>
                          <div className="aspect-square rounded-[32px] overflow-hidden border border-white/5 bg-black shadow-2xl animate-glow">
                            {result.gradcam_image ? (
                              <img
                                src={`data:image/jpeg;base64,${result.gradcam_image}`}
                                className="w-full h-full object-cover"
                                alt="Grad-CAM"
                              />
                            ) : (
                              <div className="w-full h-full flex flex-col items-center justify-center text-slate-600 gap-2">
                                <X className="w-8 h-8" />
                                <p className="text-xs">Visualization unavailable</p>
                              </div>
                            )}
                          </div>
                        </div>
                      </div>

                      {/* Report Section */}
                      <div className="space-y-6">
                        <div className="flex items-center justify-between">
                          <div>
                            <h3 className="text-2xl font-bold text-white mb-1">Diagnosis Intelligence</h3>
                            <p className="text-xs text-slate-500 underline decoration-brand-primary/30 underline-offset-4">RAG-Enhanced Clinical Analysis</p>
                          </div>
                          <button
                            onClick={downloadPDF}
                            className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-white/5 hover:bg-brand-primary/10 border border-white/10 hover:border-brand-primary/30 transition-all text-xs font-bold group shadow-xl shadow-black/20"
                          >
                            <Download className="w-3.5 h-3.5 group-hover:translate-y-0.5 transition-transform" />
                            Download PDF Report
                          </button>
                        </div>

                        <div className="grid gap-4">
                          {(() => {
                            const sections = [];
                            let currentSection = null;

                            result.report.split('\n').forEach(line => {
                              if (line.startsWith('## ')) {
                                if (currentSection) sections.push(currentSection);
                                currentSection = { title: line.replace('## ', ''), content: [] };
                              } else if (currentSection && line.trim() !== '' && !line.startsWith('# ')) {
                                currentSection.content.push(line);
                              }
                            });
                            if (currentSection) sections.push(currentSection);

                            const iconMap = {
                              "Patient/Case Information": <Activity className="w-5 h-5 text-emerald-400" />,
                              "Explainable AI (XAI) Findings": <Zap className="w-5 h-5 text-brand-primary" />,
                              "Clinical Context (Retrieved via RAG)": <Brain className="w-5 h-5 text-brand-secondary" />,
                              "Summary and Discussion": <CheckCircle2 className="w-5 h-5 text-blue-400" />
                            };


                            return sections.map((sec, idx) => (
                              <motion.div
                                key={idx}
                                initial={{ opacity: 0, y: 10 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ delay: 0.1 * idx }}
                                className="p-6 rounded-3xl glass border border-white/5 hover:border-white/10 transition-colors group relative overflow-hidden"
                              >
                                <div className="absolute top-0 right-0 p-4 opacity-[0.03] group-hover:opacity-[0.07] transition-opacity">
                                  {iconMap[sec.title.trim()] || <FileText className="w-12 h-12" />}
                                </div>
                                <div className="flex items-center gap-3 mb-4">
                                  <div className="p-2 rounded-lg bg-white/5 border border-white/10">
                                    {iconMap[sec.title.trim()] || <FileText className="w-4 h-4 text-slate-400" />}
                                  </div>
                                  <h4 className="font-bold text-slate-200 text-sm tracking-wide">{sec.title}</h4>
                                </div>
                                <div className="space-y-3 px-1">
                                  {sec.content.map((c, i) => (
                                    <p key={i} className="text-slate-400 text-xs leading-relaxed">
                                      {c.startsWith('- ') ? (
                                        <span className="flex gap-2">
                                          <span className="text-brand-primary mt-1">•</span>
                                          <span>{renderFormattedText(c.replace('- ', ''))}</span>
                                        </span>
                                      ) : renderFormattedText(c)}
                                    </p>
                                  ))}
                                </div>
                              </motion.div>
                            ));
                          })()}
                        </div>

                        <div className="p-6 rounded-2xl bg-white/[0.02] border border-white/5 text-center">
                          <p className="text-[10px] text-slate-600 italic">
                            Disclaimer: This report is generated by an AI assistant for research purposes and should be reviewed by a qualified medical professional.
                          </p>
                        </div>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            </motion.div>
          )}

          {currentView === 'dashboard' && (
            <motion.div
              key="dashboard"
              initial={{ opacity: 0, x: 50 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -50 }}
              className="space-y-12"
            >
              <div className="text-center space-y-4 max-w-2xl mx-auto mb-12">
                <h2 className="text-4xl font-black text-white">Model Performance Dashboard</h2>
                <p className="text-slate-400 text-sm">Comparative analysis of our neural architectures based on the thesis dataset benchmarks.</p>
              </div>

              <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
                {[
                  { name: 'CNN+CLAHE', acc: '94.2%', prec: '93.8%', rec: '94.5%', color: 'from-emerald-500 to-teal-500' },
                  { name: 'ResNet50', acc: '92.5%', prec: '91.2%', rec: '93.8%', color: 'from-blue-500 to-indigo-500' },
                  { name: 'VGG16', acc: '89.7%', prec: '88.5%', rec: '90.9%', color: 'from-purple-500 to-pink-500' },
                  { name: 'VGG19', acc: '90.4%', prec: '89.8%', rec: '91.0%', color: 'from-orange-500 to-red-500' },
                ].map((m) => (
                  <div key={m.name} className="glass p-8 rounded-[32px] border border-white/5 hover:border-white/10 transition-all group">
                    <div className={`w-12 h-12 rounded-2xl bg-gradient-to-br ${m.color} flex items-center justify-center mb-6 shadow-lg`}>
                      <Activity className="text-white w-6 h-6" />
                    </div>
                    <h3 className="text-xl font-bold text-white mb-4">{m.name}</h3>
                    <div className="space-y-4">
                      <div className="flex justify-between items-center">
                        <span className="text-xs text-slate-500 uppercase font-bold tracking-widest">Accuracy</span>
                        <span className="text-sm font-black text-white">{m.acc}</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-xs text-slate-500 uppercase font-bold tracking-widest">Precision</span>
                        <span className="text-sm font-black text-slate-300">{m.prec}</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-xs text-slate-500 uppercase font-bold tracking-widest">Recall</span>
                        <span className="text-sm font-black text-slate-300">{m.rec}</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>

              <div className="glass p-12 rounded-[40px] border border-white/5">
                <h3 className="text-2xl font-bold text-white mb-8">Metrics Interpretation</h3>
                <div className="grid md:grid-cols-3 gap-12 font-medium">
                  <div className="space-y-3">
                    <div className="flex items-center gap-2 text-emerald-400">
                      <CheckCircle2 className="w-4 h-4" />
                      <span className="text-sm font-bold uppercase tracking-widest">Overall Leader</span>
                    </div>
                    <p className="text-slate-400 text-sm leading-relaxed">
                      {renderFormattedText("The **CNN+CLAHE** configuration consistently outperforms others, demonstrating the massive impact of local contrast enhancement on mammography feature extraction.")}
                    </p>
                  </div>
                  <div className="space-y-3">
                    <div className="flex items-center gap-2 text-blue-400">
                      <Brain className="w-4 h-4" />
                      <span className="text-sm font-bold uppercase tracking-widest">Generalization</span>
                    </div>
                    <p className="text-slate-400 text-sm leading-relaxed">
                      {renderFormattedText("**ResNet50** shows higher variance but superior generalization on cross-institutional datasets, likely due to its deeper residual architecture.")}
                    </p>
                  </div>
                  <div className="space-y-3">
                    <div className="flex items-center gap-2 text-brand-primary">
                      <Zap className="w-4 h-4" />
                      <span className="text-sm font-bold uppercase tracking-widest">Feature Focus</span>
                    </div>
                    <p className="text-slate-400 text-sm leading-relaxed">
                      {renderFormattedText("**VGG Families** excel at detecting micro-calcifications but often suffer from vanishing gradients in purely local feature identification compared to CNN+CLAHE.")}
                    </p>
                  </div>
                </div>
              </div>
            </motion.div>
          )}

          {currentView === 'documentation' && (
            <motion.div
              key="documentation"
              initial={{ opacity: 0, x: 50 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -50 }}
              className="max-w-4xl mx-auto space-y-16"
            >
              <div className="text-center space-y-4">
                <h2 className="text-4xl font-black text-white">Technical Documentation</h2>
                <p className="text-slate-400">Understanding the core technologies powering MedVision AI.</p>
              </div>

              <div className="space-y-24">
                <section className="grid md:grid-cols-2 gap-12 items-center">
                  <div className="space-y-6">
                    <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-brand-primary/10 border border-brand-primary/20 text-brand-primary text-[10px] font-bold uppercase tracking-widest">
                      Pre-processing
                    </div>
                    <h3 className="text-2xl font-bold text-white">CLAHE Optimization</h3>
                    <p className="text-slate-400 leading-relaxed">
                      Contrast Limited Adaptive Histogram Equalization (CLAHE) is used to enhance the visibility of micro-calcifications and architectural distortions in mammograms.
                      Classic histogram equalization often over-amplifies noise in homogeneous regions; CLAHE limits this contrast enhancement through local clipping.
                    </p>
                  </div>
                  <div className="p-4 glass rounded-[32px] border border-white/5 bg-gradient-to-br from-brand-primary/5 to-transparent">
                    <img
                      src="/illustrations/clahe.png"
                      alt="CLAHE Process Visualization"
                      className="w-full h-full object-cover rounded-2xl shadow-2xl border border-white/5"
                    />
                  </div>
                </section>

                <section className="grid md:grid-cols-2 gap-12 items-center direction-rtl">
                  <div className="p-4 glass rounded-[32px] border border-white/5 bg-gradient-to-br from-brand-secondary/5 to-transparent order-2 md:order-1">
                    <img
                      src="/illustrations/gradcam.png"
                      alt="Grad-CAM Heatmap Generation"
                      className="w-full h-full object-cover rounded-2xl shadow-2xl border border-white/5"
                    />
                  </div>
                  <div className="space-y-6 order-1 md:order-2">
                    <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-brand-secondary/10 border border-brand-secondary/20 text-brand-secondary text-[10px] font-bold uppercase tracking-widest">
                      Visualization
                    </div>
                    <h3 className="text-2xl font-bold text-white">Explainable AI (Grad-CAM)</h3>
                    <p className="text-slate-400 leading-relaxed">
                      Gradient-weighted Class Activation Mapping (Grad-CAM) uses the gradients of any target concept flowing into the final convolutional layer to produce a coarse localization map highlighting the important regions in the image for predicting the concept.
                    </p>
                  </div>
                </section>

                <section className="grid md:grid-cols-2 gap-12 items-center">
                  <div className="space-y-6">
                    <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-[10px] font-bold uppercase tracking-widest">
                      Knowledge Retrieval
                    </div>
                    <h3 className="text-2xl font-bold text-white">RAG-Enhanced Reporting</h3>
                    <p className="text-slate-400 leading-relaxed">
                      Our system doesn't just predict; it references. Retrieval-Augmented Generation (RAG) pulls relevant clinical context from a medical knowledge base based on the specific anatomical findings detected by the CNN. This ensures our reports are grounded in clinical literature.
                    </p>
                  </div>
                  <div className="p-4 glass rounded-[32px] border border-white/5 bg-gradient-to-br from-emerald-500/5 to-transparent">
                    <img
                      src="/illustrations/rag.png"
                      alt="RAG Architecture Diagram"
                      className="w-full h-full object-cover rounded-2xl shadow-2xl border border-white/5"
                    />
                  </div>
                </section>
              </div>
            </motion.div>
          )}

          {currentView === 'log' && (
            <motion.div
              key="log"
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -30 }}
              className="space-y-8"
            >
              <div className="flex items-end justify-between">
                <div className="space-y-1">
                  <h2 className="text-3xl font-black text-white">Analysis Log</h2>
                  <p className="text-slate-500 text-sm">Session history of neural diagnoses performed.</p>
                </div>
                <button
                  onClick={() => setAnalysisHistory([])}
                  className="px-4 py-2 rounded-xl bg-white/5 hover:bg-red-500/10 border border-white/10 hover:border-red-500/30 text-red-500 text-[10px] font-bold uppercase tracking-widest transition-all"
                >
                  Clear History
                </button>
              </div>

              {analysisHistory.length === 0 ? (
                <div className="h-[400px] glass rounded-[40px] border border-dashed border-white/10 flex flex-col items-center justify-center text-center p-12">
                  <FileText className="w-16 h-16 text-white/5 mb-4" />
                  <h3 className="text-xl font-bold text-slate-600">No History Recorded</h3>
                  <p className="text-sm text-slate-700 max-w-xs mt-2">Diagnoses performed in this session will appear here for comparison and review.</p>
                </div>
              ) : (
                <div className="overflow-hidden glass rounded-[32px] border border-white/5">
                  <table className="w-full text-left border-collapse">
                    <thead>
                      <tr className="border-b border-white/5 bg-white/[0.02]">
                        <th className="px-8 py-6 text-[10px] font-black uppercase tracking-[0.2em] text-slate-500">Image Preview</th>
                        <th className="px-8 py-6 text-[10px] font-black uppercase tracking-[0.2em] text-slate-500">Timestamp</th>
                        <th className="px-8 py-6 text-[10px] font-black uppercase tracking-[0.2em] text-slate-500">Model Used</th>
                        <th className="px-8 py-6 text-[10px] font-black uppercase tracking-[0.2em] text-slate-500">Diagnosis</th>
                        <th className="px-8 py-6 text-[10px] font-black uppercase tracking-[0.2em] text-slate-500 text-right">Confidence</th>
                      </tr>
                    </thead>
                    <tbody>
                      {analysisHistory.map((item) => (
                        <tr key={item.id} className="border-b border-white/5 hover:bg-white/[0.01] transition-colors group">
                          <td className="px-8 py-6">
                            <div className="w-16 h-16 rounded-xl overflow-hidden border border-white/10 bg-black group-hover:scale-110 transition-transform">
                              <img src={item.preview} className="w-full h-full object-cover" alt="Scan" />
                            </div>
                          </td>
                          <td className="px-8 py-6">
                            <span className="text-xs font-bold text-slate-400 uppercase">{item.timestamp}</span>
                          </td>
                          <td className="px-8 py-6">
                            <div className="flex items-center gap-2">
                              <div className="w-1.5 h-1.5 rounded-full bg-brand-primary" />
                              <span className="text-xs font-bold text-white">{item.model}</span>
                            </div>
                          </td>
                          <td className="px-8 py-6">
                            <span className={`px-3 py-1 rounded-full text-[10px] font-black uppercase border ${item.label === 'Cancer'
                              ? 'bg-red-500/10 border-red-500/20 text-red-500'
                              : 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400'
                              }`}>
                              {item.label}
                            </span>
                          </td>
                          <td className="px-8 py-6 text-right">
                            <span className="text-sm font-black text-slate-200">{item.confidence.toFixed(1)}%</span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      <footer className="max-w-7xl mx-auto px-6 py-12 border-t border-white/5 flex flex-col items-center gap-4">
        <div className="flex items-center gap-6 text-slate-500">
          <Activity className="w-5 h-5 opacity-50" />
          <div className="w-1.5 h-1.5 rounded-full bg-slate-800" />
          <p className="text-sm font-medium">黃世漢 - Darryl</p>
          <div className="w-1.5 h-1.5 rounded-full bg-slate-800" />
          <p className="text-sm font-medium tracking-tight">© 2026</p>
        </div>
      </footer>
    </div>
  );
}

export default App;
