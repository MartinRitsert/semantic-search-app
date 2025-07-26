"use client";

import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import { 
  TypingIndicator, 
  UploadIcon, 
  SendIcon, 
  CheckIcon, 
  WarningIcon, 
  ClockIcon, 
  LoadingSpinner 
} from '@/components/Icons';

// --- Type Definitions ---
/**
 * Defines the structure for a single message in the chat history.
 */
interface Message {
  role: 'user' | 'model';
  text: string;
}

// --- Main Chat Component ---
/**
 * The main chat component that handles the entire user interface,
 * state management, and API interactions for the RAG application.
 */
export default function ChatPage() {
  // State for the user ID
  const [userId, setUserId] = useState<string | null>(null);

  // State for the conversation history
  const [messages, setMessages] = useState<Message[]>([]);

  // State for the user's current input
  const [query, setQuery] = useState<string>('');
  
  // State for the current conversation's session ID
  const [chatSessionId, setChatSessionId] = useState<string | null>(null);

  // State for loading and status indicators
  const [queryState, setQueryState] = useState({ isLoading: false, error: null as string | null });
  const [uploadState, setUploadState] = useState({
    status: 'idle',  // 'idle' | 'processing' | 'success' | 'failed'
    message: '',
    documentId: null as string | null,
    filename: null as string | null,
  });

  // Refs for direct DOM element access
  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // State for managing the "taking longer" message
  const [showTakingLonger, setShowTakingLonger] = useState<boolean>(false);

  // Effect to get or create a user ID on first load
  useEffect(() => {
    const getOrCreateUser = async () => {
      let existingUserId = localStorage.getItem('ragAppUserId');
      if (existingUserId) {
        setUserId(existingUserId);
      } else {
        const apiUrl = `${process.env.NEXT_PUBLIC_API_URL}/users/`;
        const response = await fetch(apiUrl, { method: 'POST' });
        const data = await response.json();
        localStorage.setItem('ragAppUserId', data.user_id);
        setUserId(data.user_id);
      }
    };
    getOrCreateUser();
  }, []);

  // Effect to auto-scroll to the latest message
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, queryState.isLoading]);

  // Polling for index readiness when in 'processing' state
  useEffect(() => {
    if (uploadState.status !== 'processing' || !uploadState.documentId) {
      return;
    }

    // Timer for the "taking longer" message
    const takingLongerTimeout = setTimeout(() => {
      setShowTakingLonger(true);
    }, 16000);

    const pollInterval = 2000;  // Poll every 2 seconds

    const intervalId = setInterval(async () => {
      try {
        const apiUrl = `${process.env.NEXT_PUBLIC_API_URL}/upload/status/${uploadState.documentId}`;
        const response = await fetch(apiUrl);

        if (!response.ok) throw new Error('Status check failed');

        const data = await response.json(); // e.g., {status: 'success', message: '...'}

        // If the task is done (either success or fail), stop polling and update the UI.
        if (data.status === 'success' || data.status === 'failed') {
          clearInterval(intervalId);
          setShowTakingLonger(false); 
          setUploadState(prev => ({ ...prev, status: data.status, message: data.message || data.error }));
        }
        // If the status is still 'processing', the interval will just continue.

      } catch (error) {
        console.error("Polling error:", error);
        clearInterval(intervalId);
        setShowTakingLonger(false);
        setUploadState(prev => ({ ...prev, status: 'failed', message: 'Error checking processing status.' }));
      }
    }, pollInterval);

    // Cleanup function: important to clear both timeouts
    return () => {
      clearInterval(intervalId);
      clearTimeout(takingLongerTimeout);
    };
  }, [uploadState.status, uploadState.documentId]);

  // Handler for file input changes
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && file.type === "application/pdf") {
      // Clear previous errors
      setUploadState({
        status: 'idle',
        message: '',
        documentId: null,
        filename: null
      });
      handleFileUpload(file);
    } else {
      if (file) {
        console.warn('Invalid file type:', file.type);
        setUploadState({
          status: 'error',
          message: 'Please select a valid PDF file.',
          documentId: null,
          filename: null
        });
      }
    }
    // Reset file input to allow re-uploading the same file
    if (e.target) {
      e.target.value = '';
    }
  };

  // Handler for the "Upload PDF" button click
  const handleUploadClick = () => {
    // Show a simple, informative alert to the user.
    const warningMessage = "Public Demo Notice:\n\n" +
                      "For this demo, your document will be processed and stored temporarily.\n\n" +
                      "Please do not upload any private or sensitive information.";
    window.alert(warningMessage);
    
    // Programmatically click the hidden file input to open the file browser.
    fileInputRef.current?.click();
  };

  // Handler for submitting the selected file to the backend
  const handleFileUpload = async (file: File) => {
    if (!userId) {
      console.error('User ID is not set. Cannot upload file.');
      setUploadState({
        status: 'failed',
        message: 'User session not initialized. Please refresh.',
        documentId: null,
        filename: null
      });
      return;
    }

    setShowTakingLonger(false);

    // Reset chat messages and session ID
    setMessages([]);
    setChatSessionId(null);

    // Instantly update the UI to show processing has started.
    setUploadState({
      status: 'processing',
      message: `Processing "${file.name}"...`,
      documentId: null,
      filename: file.name
    });

    const formData = new FormData();
    formData.append('file', file);
    formData.append('user_id', userId);

    try {
      const apiUrl = `${process.env.NEXT_PUBLIC_API_URL}/upload/`;
      const response = await fetch(apiUrl, { method: 'POST', body: formData });
      
      if (response.status !== 202) {  // Expecting a 202 Accepted status
          const errorData = await response.json();
          throw new Error(errorData.detail || "File upload failed.");
      }

      const data = await response.json();

      setUploadState(prev => ({ ...prev, documentId: data.document_id }));

    } catch (err: unknown) {
      console.error('Error during file upload:', err);

      // Handle failures of the initial request.
      let errorMessage = 'Upload failed. Please try again.';
      if (err instanceof Error) {
        errorMessage = err.message;
      }
      
      setUploadState({
        status: 'failed',
        message: errorMessage,
        documentId: null,
        filename: null
      });
    }
  };

  // Handler for submitting the user's query to the backend
  const handleSubmitQuery = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!query.trim()) return;

    console.info('Submitting query:', query);

    // Update the UI with the user's message
    setMessages(prev => [...prev, { role: 'user', text: query }]);
    setQueryState({ isLoading: true, error: null });
    setQuery('');

    try {
      const apiUrl = `${process.env.NEXT_PUBLIC_API_URL}/query/`;
      const response = await fetch(apiUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: query,
          chat_session_id: chatSessionId,
          document_id: uploadState.documentId,
          user_id: userId,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'An unexpected error occurred while processing your request.');
      }

      const data = await response.json();
      console.info('Query response received:', data);
      setMessages(prev => [...prev, { role: 'model', text: data.answer }]);
      setChatSessionId(data.chat_session_id);

    } catch (err: unknown) {
      console.error('Error during query submission:', err);

      let errorMessage = 'An unexpected error occurred.';
      if (err instanceof Error) {
        errorMessage = err.message;
      }

      setQueryState(prev => ({ ...prev, error: errorMessage }));
      setMessages(prev => [...prev, { role: 'model', text: errorMessage }]);
    } finally {
      setQueryState(prev => ({ ...prev, isLoading: false }));
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gray-900 text-gray-200 font-sans">
      <header className="p-4 md:p-6 border-b border-gray-700/60 bg-gray-900/50 backdrop-blur-sm">
        <div className="max-w-5xl mx-auto flex items-center justify-between">
            <div className="flex flex-col">
                <h1 className="text-xl md:text-2xl font-bold bg-gradient-to-r from-blue-500 to-purple-500 bg-clip-text text-transparent">
                    RAG Assistant
                </h1>
                <p className="text-xs md:text-sm text-gray-400 tracking-wide">
                    Built with FastAPI, Next.js, Pinecone and Gemini
                </p>
            </div>
            <div className="flex items-center space-x-3 flex-1 justify-end">
                <input
                    type="file"
                    ref={fileInputRef}
                    onChange={handleFileChange}
                    accept="application/pdf"
                    className="hidden"
                    disabled={uploadState.status === 'processing'}
                />
                <button
                  onClick={handleUploadClick}
                  disabled={uploadState.status === 'processing'}
                  className="flex items-center justify-center space-x-2 px-4 py-2 
                            bg-gray-700 rounded-lg hover:bg-gray-600 transition-colors w-[150px] h-[40px] whitespace-nowrap 
                            disabled:bg-gray-800 disabled:text-gray-400 disabled:cursor-not-allowed"
                >
                  <>
                    <UploadIcon />
                    <span>Upload PDF</span>
                  </>
                </button>
            </div>
        </div>
        {uploadState.message && uploadState.status !== 'idle' && (
            <div className="max-w-5xl mx-auto mt-4">
              {(uploadState.status === 'processing' || showTakingLonger) ? (
                <div
                  className={`relative rounded-lg p-0.5 animate-border-rotate ${
                    showTakingLonger
                      ? 'bg-[conic-gradient(from_var(--border-angle)_at_50%_50%,#f59e0b_0%,#f97316_50%,#f59e0b_100%)]'
                      : 'bg-[conic-gradient(from_var(--border-angle)_at_50%_50%,#3b82f6_0%,#a855f7_50%,#3b82f6_100%)]'
                  }`}
                >
                  <div className={`flex items-center space-x-3 p-3 rounded-[7px] bg-gray-900 text-sm ${
                    showTakingLonger ? 'text-orange-300' : 'text-gray-300'
                  }`}>
                    <ClockIcon />
                    <span>
                      {showTakingLonger
                        ? "Processing is taking longer than usual... You can ask questions shortly."
                        : uploadState.message}
                    </span>
                  </div>
                </div>
              ) : (
                <div 
                    className={`flex items-center space-x-3 p-3 rounded-lg border text-sm shadow-lg
                        ${
                          {
                            'success': 'bg-green-900/20 border-green-500/30 text-green-300 shadow-[0_0_15px_rgba(34,197,94,0.2)]',
                            'failed': 'bg-red-900/20 border-red-500/30 text-red-300 shadow-[0_0_15px_rgba(239,68,68,0.2)]'
                          }[uploadState.status]
                        }
                    `}
                >
                    {uploadState.status === 'success' && <CheckIcon />}
                    {uploadState.status === 'failed' && <WarningIcon />}
                    <span>{uploadState.message}</span>
                </div>
              )}
            </div>
        )}
      </header>
      <main className="flex-1 overflow-y-auto p-4 md:p-6">
        <div className="max-w-4xl mx-auto">
          {messages.length === 0 && (
             <div className="text-center text-gray-500 mt-10">
                <p>No messages yet.</p>
                <p>
                  {uploadState.status === 'success'
                    ? "PDF is ready - Start asking questions!"
                    : "Upload a PDF to get started!"}
                </p>
             </div>
          )}
          {messages.map((message, index) => (
            <div
              key={index}
              className={`flex mb-4 ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`rounded-2xl p-3 max-w-lg lg:max-w-2xl shadow-lg animate-fade-in ${
                  message.role === 'user'
                    ? 'bg-blue-600 text-white rounded-br-none'
                    : 'bg-gray-700 text-gray-200 rounded-bl-none'
                }`}
              >
                <div>
                  <ReactMarkdown
                    components={{
                      ul: ({ node, ...props }) => <ul className="list-disc pl-5" {...props} />,
                      ol: ({ node, ...props }) => <ol className="list-decimal pl-5" {...props} />,
                    }}
                  >
                    {message.text}
                  </ReactMarkdown>
                </div>
              </div>
            </div>
          ))}
          {queryState.isLoading && (
            <div className="flex mb-4 justify-start">
              <div className="rounded-2xl p-4 max-w-lg lg:max-w-2xl shadow-lg animate-fade-in bg-gray-700 text-gray-200 rounded-bl-none">
                <TypingIndicator />
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </main>

      <footer className="bg-gray-800/50 backdrop-blur-sm p-4 border-t border-gray-700">
        <div className="max-w-4xl mx-auto">
          <form onSubmit={handleSubmitQuery} className="flex items-center space-x-2">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder={
                uploadState.status === 'success'
                  ? "Ask a question about the uploaded document..."
                  : "Upload a PDF before asking questions!"
              }
              className="flex-1 p-3 bg-gray-700 rounded-full focus:outline-none focus:ring-2 focus:ring-blue-500 text-gray-100 border border-transparent"
            />
            <button
              type="submit"
              className="bg-blue-600 p-3 rounded-full hover:bg-blue-700 disabled:bg-blue-800 disabled:cursor-not-allowed transition-colors flex items-center justify-center w-12 h-12 shrink-0"
              disabled={
                uploadState.status !== 'success' ||
                queryState.isLoading ||
                !query.trim()
              }
            >
              {queryState.isLoading ? <LoadingSpinner /> : <SendIcon />}
            </button>
          </form>
        </div>
      </footer>
    </div>
  );
}
