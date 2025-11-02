import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Search, Image, Loader2, AlertCircle, ZoomIn } from 'lucide-react';

const ImageVisionUI = () => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [selectedImage, setSelectedImage] = useState(null);
  const [embeddings, setEmbeddings] = useState(null);
  const [isInitialized, setIsInitialized] = useState(false);
  const debounceTimer = useRef(null);

  useEffect(() => {
    initializeSearch();
  }, []);

  const initializeSearch = async () => {
    try {
      setIsLoading(true);
      
      const response = await fetch('http://localhost:5000/api/stats');
      const data = await response.json();
      
      if (data.success) {
        setEmbeddings(data);
        setIsInitialized(true);
      } else {
        throw new Error('Failed to load stats');
      }
    } catch (err) {
      setError('Failed to connect to backend. Make sure Flask server is running on port 5000.');
      console.error(err);
      // Set initialized anyway to show the UI
      setIsInitialized(true);
    } finally {
      setIsLoading(false);
    }
  };

  const performSearch = useCallback(async (searchQuery) => {
    if (!searchQuery.trim()) {
      setResults([]);
      return;
    }

    setIsLoading(true);
    setError('');

    try {
      const response = await fetch('http://localhost:5000/api/search', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: searchQuery,
          top_k: 12
        })
      });

      if (!response.ok) {
        throw new Error('Search request failed');
      }

      const data = await response.json();
      
      const resultsWithThumbnails = data.results.map(result => ({
        ...result,
        thumbnail: `http://localhost:5000/api/image/${result.filename}`
      }));

      setResults(resultsWithThumbnails);
    } catch (err) {
      setError('Search failed. Make sure the backend server is running on localhost:5000');
      console.error(err);
      setResults([]);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const handleSearchChange = (e) => {
    const value = e.target.value;
    setQuery(value);

    if (debounceTimer.current) {
      clearTimeout(debounceTimer.current);
    }

    debounceTimer.current = setTimeout(() => {
      performSearch(value);
    }, 300);
  };

  const handleImageClick = (result) => {
    setSelectedImage(result);
  };

  const closeModal = () => {
    setSelectedImage(null);
  };

  if (!isInitialized) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-white to-purple-50 flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="w-12 h-12 animate-spin text-indigo-600 mx-auto mb-4" />
          <p className="text-gray-600">Initializing Image Vision...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-white to-purple-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <div className="flex items-center gap-3 mb-6">
            <div className="bg-gradient-to-br from-indigo-500 to-purple-600 p-2 rounded-lg">
              <Image className="w-8 h-8 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent">
                Image Vision
              </h1>
              <p className="text-sm text-gray-500">Semantic Image Search</p>
            </div>
            {embeddings && embeddings.total_images && (
              <div className="ml-auto bg-indigo-50 px-4 py-2 rounded-full">
                <span className="text-sm font-medium text-indigo-700">
                  {embeddings.total_images} images indexed
                </span>
              </div>
            )}
          </div>

          {/* Search Bar */}
          <div className="relative">
            <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
            <input
              type="text"
              value={query}
              onChange={handleSearchChange}
              placeholder="Describe what you're looking for... (e.g., 'red flowers on table')"
              className="w-full pl-12 pr-4 py-4 text-lg border-2 border-gray-200 rounded-xl focus:border-indigo-500 focus:ring-4 focus:ring-indigo-100 outline-none transition-all"
              autoFocus
            />
            {isLoading && (
              <Loader2 className="absolute right-4 top-1/2 transform -translate-y-1/2 text-indigo-500 w-5 h-5 animate-spin" />
            )}
          </div>

          {/* Stats */}
          {query && (
            <div className="mt-3 flex items-center gap-4 text-sm text-gray-600">
              <span className="font-medium">{results.length} results found</span>
              {results.length > 0 && (
                <>
                  <span className="text-gray-400">•</span>
                  <span>Best match: {(results[0].score * 100).toFixed(1)}% similarity</span>
                </>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 py-8">
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6 flex items-center gap-3">
            <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0" />
            <div className="flex-1">
              <p className="text-red-700 font-medium">{error}</p>
              <p className="text-red-600 text-sm mt-1">Run: python backend_api.py</p>
            </div>
          </div>
        )}

        {/* Empty State */}
        {!query && !isLoading && (
          <div className="text-center py-20">
            <div className="bg-gradient-to-br from-indigo-100 to-purple-100 w-24 h-24 rounded-full flex items-center justify-center mx-auto mb-6">
              <Search className="w-12 h-12 text-indigo-600" />
            </div>
            <h2 className="text-2xl font-semibold text-gray-800 mb-2">
              Start Your Search
            </h2>
            <p className="text-gray-600 max-w-md mx-auto mb-8">
              Enter a description to find images using AI-powered semantic search
            </p>
            <div className="flex flex-wrap justify-center gap-2">
              {['red flowers', 'person with glasses', 'wooden furniture', 'coffee mug', 'blue wall', 'green plants'].map((suggestion) => (
                <button
                  key={suggestion}
                  onClick={() => {
                    setQuery(suggestion);
                    performSearch(suggestion);
                  }}
                  className="px-4 py-2 bg-white border border-gray-200 rounded-full text-sm text-gray-600 hover:border-indigo-300 hover:text-indigo-600 hover:shadow-md transition-all"
                >
                  {suggestion}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* No Results */}
        {query && results.length === 0 && !isLoading && !error && (
          <div className="text-center py-20">
            <div className="bg-gray-100 w-24 h-24 rounded-full flex items-center justify-center mx-auto mb-6">
              <Image className="w-12 h-12 text-gray-400" />
            </div>
            <h2 className="text-2xl font-semibold text-gray-800 mb-2">
              No Results Found
            </h2>
            <p className="text-gray-600">
              Try a different search term or description
            </p>
          </div>
        )}

        {/* Results Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {results.map((result, idx) => (
            <div
              key={result.filename}
              className="group bg-white rounded-xl shadow-sm hover:shadow-xl transition-all duration-300 overflow-hidden cursor-pointer border border-gray-100"
              onClick={() => handleImageClick(result)}
              style={{
                animation: `fadeIn 0.3s ease-out ${idx * 0.05}s both`
              }}
            >
              <div className="relative aspect-video bg-gray-100 overflow-hidden">
                <img
                  src={result.thumbnail}
                  alt={result.filename}
                  className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                  loading="lazy"
                  onError={(e) => {
                    e.target.src = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="400" height="300"%3E%3Crect fill="%23f3f4f6" width="400" height="300"/%3E%3Ctext x="50%25" y="50%25" text-anchor="middle" fill="%239ca3af" font-size="16"%3EImage not found%3C/text%3E%3C/svg%3E';
                  }}
                />
                <div className="absolute inset-0 bg-black opacity-0 group-hover:opacity-20 transition-opacity" />
                <div className="absolute top-3 right-3 bg-white/90 backdrop-blur-sm px-3 py-1 rounded-full text-sm font-medium text-indigo-600">
                  {(result.score * 100).toFixed(1)}%
                </div>
                <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                  <div className="bg-white/90 backdrop-blur-sm p-3 rounded-full">
                    <ZoomIn className="w-6 h-6 text-indigo-600" />
                  </div>
                </div>
              </div>
              <div className="p-4">
                <p className="text-sm text-gray-600 line-clamp-2 mb-2">
                  {result.description || 'No description available'}
                </p>
                <p className="text-xs text-gray-400 font-mono truncate">
                  {result.filename}
                </p>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Image Detail Modal */}
      {selectedImage && (
        <div
          className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4"
          onClick={closeModal}
        >
          <div
            className="bg-white rounded-2xl max-w-4xl w-full max-h-[90vh] overflow-auto"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="p-6">
              <div className="flex justify-between items-start mb-4">
                <div>
                  <h3 className="text-2xl font-bold text-gray-800 mb-1">
                    Image Details
                  </h3>
                  <p className="text-sm text-gray-500 font-mono">
                    {selectedImage.filename}
                  </p>
                </div>
                <button
                  onClick={closeModal}
                  className="text-gray-400 hover:text-gray-600 text-3xl leading-none px-2"
                  aria-label="Close"
                >
                  ×
                </button>
              </div>
              
              <img
                src={selectedImage.thumbnail}
                alt={selectedImage.filename}
                className="w-full rounded-lg mb-4 max-h-96 object-contain bg-gray-50"
              />
              
              <div className="bg-gray-50 rounded-lg p-4">
                <h4 className="font-semibold text-gray-800 mb-2">Description</h4>
                <p className="text-gray-600 mb-4">{selectedImage.description || 'No description available'}</p>
                
                <div className="flex items-center gap-3 mb-4">
                  <span className="text-sm text-gray-500 min-w-fit">Similarity Score:</span>
                  <div className="flex-1 bg-gray-200 rounded-full h-2.5 max-w-xs">
                    <div
                      className="bg-gradient-to-r from-indigo-500 to-purple-600 h-2.5 rounded-full transition-all"
                      style={{ width: `${selectedImage.score * 100}%` }}
                    />
                  </div>
                  <span className="text-sm font-semibold text-indigo-600 min-w-fit">
                    {(selectedImage.score * 100).toFixed(1)}%
                  </span>
                </div>

                {selectedImage.full_data && (
                  <div className="mt-4 pt-4 border-t border-gray-200">
                    <button
                      onClick={() => {
                        navigator.clipboard.writeText(JSON.stringify(selectedImage.full_data, null, 2));
                        alert('Full data copied to clipboard!');
                      }}
                      className="text-sm text-indigo-600 hover:text-indigo-700 font-medium"
                    >
                      Copy full metadata →
                    </button>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      <style>{`
        @keyframes fadeIn {
          from {
            opacity: 0;
            transform: translateY(20px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        .line-clamp-2 {
          display: -webkit-box;
          -webkit-line-clamp: 2;
          -webkit-box-orient: vertical;
          overflow: hidden;
        }
      `}</style>
    </div>
  );
};

export default ImageVisionUI;