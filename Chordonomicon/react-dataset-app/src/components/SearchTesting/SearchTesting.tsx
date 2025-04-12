import React, { useState } from 'react';

const SearchTesting: React.FC = () => {
    const [searchTerm, setSearchTerm] = useState('');
    const [results, setResults] = useState<string[]>([]);

    const handleSearch = () => {
        // Implement search logic here
        // For demonstration, we'll just filter a static dataset
        const dataset = ['Dataset 1', 'Dataset 2', 'Dataset 3'];
        const filteredResults = dataset.filter(item => 
            item.toLowerCase().includes(searchTerm.toLowerCase())
        );
        setResults(filteredResults);
    };

    return (
        <div>
            <h2>Search Datasets</h2>
            <input 
                type="text" 
                value={searchTerm} 
                onChange={(e) => setSearchTerm(e.target.value)} 
                placeholder="Enter search term" 
            />
            <button onClick={handleSearch}>Search</button>
            <ul>
                {results.map((result, index) => (
                    <li key={index}>{result}</li>
                ))}
            </ul>
        </div>
    );
};

export default SearchTesting;