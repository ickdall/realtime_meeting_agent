import React, { useState, useEffect, useRef } from 'react';
import { Mic, Activity, Users, Brain, Search, MessageSquare } from 'lucide-react';

const Card = ({ children }) => (
  <div className="p-4 bg-white rounded-lg shadow-md">
    {children}
  </div>
);

const CardHeader = ({ children }) => (
  <div className="border-b pb-2 mb-2">{children}</div>
);

const CardTitle = ({ children }) => (
  <h2 className="text-lg font-bold">{children}</h2>
);

const CardContent = ({ children }) => (
  <div>{children}</div>
);

const ScrollArea = ({ children, className }) => {
  const scrollRef = useRef(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [children]);

  return (
    <div ref={scrollRef} className={`overflow-auto ${className}`}>
      {children}
    </div>
  );
};

const MonitoringDashboard = () => {
  const [realtimeTexts, setRealtimeTexts] = useState([]);
  const [summaries, setSummaries] = useState([]);
  const [qaResponses, setQaResponses] = useState([]);
  const [searchResults, setSearchResults] = useState([]);
  const [activeTab, setActiveTab] = useState('summary');
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState(null);
  const ws = useRef(null);
  const reconnectTimeoutRef = useRef(null);

  const connectWebSocket = () => {
    if (ws.current?.readyState === WebSocket.OPEN) return;

    try {
      ws.current = new WebSocket('ws://localhost:8000/ws');

      ws.current.onopen = () => {
        console.log('WebSocket Connected');
        setIsConnected(true);
        setError(null);
      };

      ws.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          if (data.type === 'text_update') {
            if (data.text) {
              setRealtimeTexts(prev => [
                ...prev,
                {
                  id: Date.now(),
                  text: data.text,
                  timestamp: new Date().toLocaleTimeString()
                }
              ]);
            }
          } else if (data.type === 'llm_update') {
            setSummaries(prev => [...prev, ...(data.results.summaries || [])]);
            setQaResponses(prev => [...prev, ...(data.results.qa_responses || [])]);
            setSearchResults(prev => [...prev, ...(data.results.search_results || [])]);
          } else if (data.type === 'initial_data') {
            setSummaries(data.llm_results.summaries || []);
            setQaResponses(data.llm_results.qa_responses || []);
            setSearchResults(data.llm_results.search_results || []);
          }
        } catch (e) {
          console.error('Error parsing WebSocket message:', e);
        }
      };

      ws.current.onclose = () => {
        console.log('WebSocket Disconnected');
        setIsConnected(false);
        reconnectTimeoutRef.current = setTimeout(connectWebSocket, 1000);
      };

      ws.current.onerror = (error) => {
        console.error('WebSocket Error:', error);
        setError('WebSocket connection error');
      };
    } catch (error) {
      console.error('Error creating WebSocket:', error);
      setError('Error creating WebSocket connection');
      reconnectTimeoutRef.current = setTimeout(connectWebSocket, 1000);
    }
  };

  useEffect(() => {
    connectWebSocket();

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (ws.current) {
        ws.current.close();
      }
    };
  }, []);

  return (
    <div className="p-4 bg-gray-50 min-h-screen">
      <div className="max-w-7xl mx-auto space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Card>
            <CardHeader>
              <CardTitle>Connection Status</CardTitle>
              <Activity className={isConnected ? "text-green-500" : "text-red-500"} />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{isConnected ? 'Connected' : 'Disconnected'}</div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Text Length</CardTitle>
              <MessageSquare className="text-purple-500" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {realtimeTexts.reduce((acc, curr) => acc + curr.text.length, 0)}
              </div>
            </CardContent>
          </Card>
        </div>

        <Card>
          <CardHeader>
            <CardTitle>Real-time Transcription</CardTitle>
            <Mic className="text-blue-500" />
          </CardHeader>
          <CardContent>
            <div className="h-96 overflow-auto rounded-md border p-4 bg-gray-50">
              <div className="space-y-4">
                <div className="text-blue-600 border-l-4 border-blue-500 pl-2">
                  {realtimeTexts.map((item, index) => (
                    <span key={`text-${index}-${item.timestamp}`}>{item.text} </span>
                  ))}
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Analysis Results</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex space-x-2 mb-4">
              {['summary', 'qa', 'search'].map((tab) => (
                <button
                  key={tab}
                  className={`px-4 py-2 rounded-lg ${
                    activeTab === tab
                      ? 'bg-blue-500 text-white'
                      : 'bg-gray-100 hover:bg-gray-200'
                  }`}
                  onClick={() => setActiveTab(tab)}
                >
                  {tab.charAt(0).toUpperCase() + tab.slice(1)}
                </button>
              ))}
            </div>

            <div className="h-60 overflow-auto rounded-md border p-4">
              {activeTab === 'summary' && (
                <div className="space-y-4">
                  {summaries.map((summary, index) => (
                    <div key={index} className="p-3 bg-white rounded-lg shadow">
                      {summary}
                    </div>
                  ))}
                </div>
              )}

              {activeTab === 'qa' && (
                <div className="space-y-4">
                  {qaResponses.map((qa, index) => (
                    <div key={index} className="p-3 bg-white rounded-lg shadow">
                      {qa}
                    </div>
                  ))}
                </div>
              )}

              {activeTab === 'search' && (
                <div className="space-y-4">
                  {searchResults.map((result, index) => (
                    <div key={index} className="p-3 bg-white rounded-lg shadow">
                      {result}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default MonitoringDashboard;