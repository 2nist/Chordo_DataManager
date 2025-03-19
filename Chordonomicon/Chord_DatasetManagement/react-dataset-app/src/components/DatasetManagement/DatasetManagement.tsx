import React, { useState } from 'react';

const DatasetManagement: React.FC = () => {
    const [datasets, setDatasets] = useState<string[]>([]);
    const [newDataset, setNewDataset] = useState<string>('');

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

    return (
        <div>
            <h2>Dataset Management</h2>
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
    );
};

export default DatasetManagement;