import express, {Request, Response, NextFunction} from "express";
import bodyParser from "body-parser";

const app = express();
const PORT = process.env.PORT || 3000;

app.use(bodyParser.json());

app.use((err: Error, req: Request, res: Response, next: NextFunction) => {
    console.error(req, err.stack);
    res.status(500).json({ message: 'Internal Server Error' });
})

app.listen(PORT, () => {
    console.log(`NextSearch listening on port: ${PORT}`);
})