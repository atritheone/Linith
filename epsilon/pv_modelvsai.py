# pv_modelvsai.py
import argparse
import torch

from linithenv import LinithEnv, SUN, MOON
from linithai import choose_hard_move
from pv_model import LinithPVNet
from pv_mcts import PV_MCTS


def play_one_game(net, device, sims, opp_difficulty, max_moves, model_as_sun: bool):
    dev = torch.device(device)
    env = LinithEnv(max_moves=max_moves)
    obs = env.reset()

    mcts = PV_MCTS(net, device=dev)

    # choose who is model
    if model_as_sun:
        model_player = SUN
    else:
        model_player = MOON

    while not env.state.done:
        s = env.state
        if s.current_player == model_player:
            # model move with MCTS
            visits = mcts.search(env, num_simulations=sims)
            action = max(visits.items(), key=lambda kv: kv[1])[0]
        else:
            # opponent: JS-style AI
            action = choose_hard_move(env, difficulty=opp_difficulty)
        obs, reward, done, info = env.step(action)

    winner = env.state.winner
    return winner, model_player


def main():
    ap = argparse.ArgumentParser(description="Evaluate PV model vs Linith AI (easy/medium/hard).")
    ap.add_argument("--model", type=str, required=True, help="PV model .pt")
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--sims", type=int, default=128)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--opp", type=str, default="medium", choices=["easy", "medium", "hard", "hard_train"])
    ap.add_argument("--max-moves", type=int, default=200)
    args = ap.parse_args()

    dev = torch.device(args.device)
    net = LinithPVNet()
    net.load_state_dict(torch.load(args.model, map_location=dev))
    net.to(dev)
    net.eval()

    model_wins = 0
    opp_wins = 0
    draws = 0

    print(f"[pv_eval] Evaluating PV model {args.model} vs {args.opp.capitalize()}")

    for i in range(args.games):
        model_as_sun = (i % 2 == 0)
        winner, model_player = play_one_game(
            net=net,
            device=args.device,
            sims=args.sims,
            opp_difficulty=args.opp,
            max_moves=args.max_moves,
            model_as_sun=model_as_sun,
        )

        if winner is None:
            draws += 1
            result_str = "draw"
        elif winner == model_player:
            model_wins += 1
            result_str = "model wins"
        else:
            opp_wins += 1
            result_str = f"{args.opp} wins"

        side = "Sun" if model_as_sun else "Moon"
        print(f"[pv_eval] game {i+1}/{args.games}: {result_str} (model was {side})")

    print("========== SUMMARY ==========")
    print(f"Games:      {args.games}")
    print(f"Model wins: {model_wins}")
    print(f"{args.opp.capitalize()} wins: {opp_wins}")
    print(f"Draws:      {draws}")


if __name__ == "__main__":
    main()
