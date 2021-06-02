import * as React from "react";
import { BrowserRouter, Route, Link, Switch } from "react-router-dom";
import { AppModel } from "../datamodel/datamodel";
import NotFound from "../pages/NotFound/NotFound";
import Viewer from "../pages/viewer/Viewer";



const AppRouter = () => {
    return (
        <React.Suspense fallback={<div>Loading...</div>}>
            <BrowserRouter>
                <Switch>
                    <Route path="/" exact component={Viewer}/>
                    <Route component={NotFound} exact/>
                </Switch>
            </BrowserRouter>
        </React.Suspense>
    );
}

export default AppRouter;