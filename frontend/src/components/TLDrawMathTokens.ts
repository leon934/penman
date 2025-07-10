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

const minus_token = minus as TLDrawShape;
const plus_token = plus as TLDrawShape;
const zero_token = zero as TLDrawShape;
const one_token = one as TLDrawShape;
const two_token = two as TLDrawShape;
const three_token = three as TLDrawShape;
const four_token = four as TLDrawShape;
const five_token = five as TLDrawShape;
const six_token = six as TLDrawShape;
const seven_token = seven as TLDrawShape;
const eight_token = eight as TLDrawShape;
const nine_token = nine as TLDrawShape;
const fwd_slash_token = fwd_slash as TLDrawShape;

export const mathTokens: Record<string, TLDrawShape> = {
    "-": minus as TLDrawShape,
    "+": plus as TLDrawShape,
    0: zero as TLDrawShape,
    1: one as TLDrawShape,
    2: two as TLDrawShape,
    3: three as TLDrawShape,
    4: four as TLDrawShape,
    5: five as TLDrawShape,
    6: six as TLDrawShape,
    7: seven as TLDrawShape,
    8: eight as TLDrawShape,
    9: nine as TLDrawShape,
    "/": fwd_slash as TLDrawShape
}