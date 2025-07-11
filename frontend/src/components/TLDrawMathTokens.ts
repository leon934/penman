import { type TLDrawShape } from "tldraw";

import minus from "../math_tokens/-.json";
import plus from "../math_tokens/+.json";
import zero from "../math_tokens/0.json";
import one from "../math_tokens/1.json";
import two from "../math_tokens/2.json";
import three from "../math_tokens/3.json";
import four from "../math_tokens/4.json";
import five from "../math_tokens/5.json";
import six from "../math_tokens/6.json";
import seven from "../math_tokens/7.json";
import eight from "../math_tokens/8.json";
import nine from "../math_tokens/9.json";
import fwd_slash from "../math_tokens/fwd_slash.json";

export const mathTokens: Record<string, TLDrawShape> = {
    "-": minus as TLDrawShape,
    "+": plus as TLDrawShape,
    "0": zero as TLDrawShape,
    "1": one as TLDrawShape,
    "2": two as TLDrawShape,
    "3": three as TLDrawShape,
    "4": four as TLDrawShape,
    "5": five as TLDrawShape,
    "6": six as TLDrawShape,
    "7": seven as TLDrawShape,
    "8": eight as TLDrawShape,
    "9": nine as TLDrawShape,
    "/": fwd_slash as TLDrawShape
}