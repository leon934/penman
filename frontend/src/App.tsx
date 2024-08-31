import React, { useEffect, useState } from 'react';

const App: React.FC = () => {
    const [data, setData] = useState<string | null>(null);

    useEffect(() => {
        console.log("test")

        fetch('http://localhost:5000/api/data')
            .then(response => response.json())
            .then(data => setData(data.message))
            .catch(error => console.error('Error fetching data:', error));
    }, []);

    return (
        <div>
            <h1>React Frontend</h1>
            <p>{data}</p>
        </div>
    );
}

export default App;
