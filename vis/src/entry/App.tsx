import * as React from 'react';
import "bootstrap/dist/css/bootstrap"
import AppRouter from './router';
import { AppModel, initialState } from '../datamodel/datamodel';
import { graphReducer } from '../datamodel/reducers';





const App = () => {

    return (
        <AppRouter/>
    );
}

export default App;