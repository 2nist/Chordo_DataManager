import React, { useState } from 'react';

import React, { useState } from 'react';

const DatasetManagement: React.FC = () => {
    const [datasets, setDatasets] = useState<string[]>([]);
    const [newDataset, setNewDataset] = useState<string>('');
    const [query, setQuery] = useState<string>('');
    const [searchResults, setSearchResults] = useState<any[]>([]);

    const addDataset = () => {
        if (newDataset) {
            setDatasets([...datasets, newDataset]);
            setNewDataset('');
        }
    };

    const editDataset = (index: number, updatedDataset: string) => {
        const updatedDatasets = datasets.map((dataset, i) => (i === index ? updatedDataset : dataset));
        setDatasets(updatedDatasets);
    };

    const deleteDataset = (index: number) => {
        const updatedDatasets = datasets.filter((_, i) => i !== index);
        setDatasets(updatedDatasets);
    };

    const handleQueryChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setQuery(e.target.value);
    };

    const handleSearch = async () => {
        try {
            const response = await fetch(`/api/query?search=${encodeURIComponent(query)}`);
            const data = await response.json();
            setSearchResults(data);
        } catch (error) {
            console.error('Error fetching query results:', error);
        }
    };

    return (
        <div>
            <h2>Dataset Management</h2>
            
            {/* Search Section */}
            <div className="search-section">
                <h3>Search Dataset</h3>
                <div className="search-input">
                    <input
                        type="text"
                        value={query}
                        onChange={handleQueryChange}
                        placeholder="Search for a song or progression..."
                    />
                    <button onClick={handleSearch}>Search</button>
                </div>
                
                <div className="search-results">
                    <h4>Search Results</h4>
                    {searchResults.length > 0 ? (
                        <ul>
                            {searchResults.map((result, index) => (
                                <li key={index}>{JSON.stringify(result)}</li>
                            ))}
                        </ul>
                    ) : (
                        <p>No results found</p>
                    )}
                </div>
            </div>
            
            {/* Existing Dataset Management UI */}
            <div className="dataset-management">
                <h3>Add New Dataset</h3>
                <input
                    type="text"
                    value={newDataset}
                    onChange={(e) => setNewDataset(e.target.value)}
                    placeholder="Add new dataset"
                />
                <button onClick={addDataset}>Add Dataset</button>
                <ul>
                    {datasets.map((dataset, index) => (
                        <li key={index}>
                            {dataset}
                            <button onClick={() => editDataset(index, prompt('Edit dataset:', dataset) || dataset)}>Edit</button>
                            <button onClick={() => deleteDataset(index)}>Delete</button>
                        </li>
                    ))}
                </ul>
            </div>
        </div>
    );
};

export default DatasetManagement;