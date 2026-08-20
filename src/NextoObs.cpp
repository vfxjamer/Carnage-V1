#include "NextoObs.h"
#include <RLGymCPP/Gamestates/StateUtil.h>
#include <mutex>
#include <cstdio>

namespace RLGC {
	constexpr float POS_COEF = 1 / 5000.f;
	constexpr float VEL_COEF = 1 / 2300.f;
	constexpr float ANG_VEL_COEF = 1 / 3.f;

	namespace {
		// Opponent block only: position + forward/up orientation + velocities
		void AddOpponentFrame(FList& obs, const PhysState& phys) {
			obs += phys.pos * POS_COEF;
			obs += phys.rotMat.forward;
			obs += phys.rotMat.up;
			obs += phys.vel * VEL_COEF;
			obs += phys.angVel * ANG_VEL_COEF;
		}
	}

	FList NextoObs::BuildObs(const Player& player, const GameState& state) {
		FList obs = {};
		obs.reserve(OBS_DIM);

		// Invert into the orange-player frame so both teams see the same view
		const bool inv = player.team == Team::ORANGE;
		const auto ball = InvertPhys(state.ball, inv);

		// Ball (18): pos(3) + rot matrix(9) + vel(3) + angVel(3)
		obs += ball.pos * POS_COEF;
		obs += ball.rotMat[0]; // forward column
		obs += ball.rotMat[1]; // right column
		obs += ball.rotMat[2]; // up column
		obs += ball.vel * VEL_COEF;
		obs += ball.angVel * ANG_VEL_COEF;

		// Previous action (8)
		for (size_t i = 0; i < Action::ELEM_AMOUNT; i++)
			obs += player.prevAction[i];

		// Boost pads (34)
		const auto& pads = state.GetBoostPads(inv);
		const auto& padTimers = state.GetBoostPadTimers(inv);
		for (int i = 0; i < CommonValues::BOOST_LOCATIONS_AMOUNT; i++) {
			if (pads[i]) {
				obs += 1.f; // Pad is already available
			} else {
				obs += 1.f / (1.f + padTimers[i]); // Approaches 1 as the pad becomes available
			}
		}

		// Self (19): phys(15) + boost/onGround/hasFlip/isDemoed(4)
		const auto selfPhys = InvertPhys(player, inv);
		obs += selfPhys.pos * POS_COEF;
		obs += selfPhys.rotMat.forward;
		obs += selfPhys.rotMat.up;
		obs += selfPhys.vel * VEL_COEF;
		obs += selfPhys.angVel * ANG_VEL_COEF;
		obs += player.boost / 100;
		obs += player.isOnGround;
		obs += player.HasFlipOrJump();
		obs += player.isDemoed;

		// Opponent (15): phys only
		for (auto& otherPlayer : state.players) {
			if (otherPlayer.carId == player.carId || otherPlayer.team == player.team)
				continue;
			AddOpponentFrame(obs, InvertPhys(otherPlayer, inv));
		}

		// Runtime verification: 1v1 must produce exactly 94 floats
		RG_ASSERT(obs.size() == OBS_DIM);

		static std::once_flag printFlag;
		std::call_once(printFlag, [] {
			printf("Observation size: %d\n", OBS_DIM);
			fflush(stdout);
		});

		return obs;
	}
}