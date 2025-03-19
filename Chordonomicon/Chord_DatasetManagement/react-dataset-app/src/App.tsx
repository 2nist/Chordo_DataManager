import React from 'react';
import DatasetManagement from './components/DatasetManagement/DatasetManagement';
import SearchTesting from './components/SearchTesting/SearchTesting';
import CustomExporting from './components/CustomExporting/CustomExporting';

const App: React.FC = () => {
  return (
    <div>
      <h1>Dataset Management Application</h1>
      <DatasetManagement />
      <SearchTesting />
      <CustomExporting />
    </div>
  );
};

export default App;