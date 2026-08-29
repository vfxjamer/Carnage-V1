#pragma once

#include <RLGymCPP/Rewards/CommonRewards.h>
#include <RLGymCPP/CommonValues.h>
#include <cmath>

namespace RLGC {

	// ============================================================
	// Speed Toward Ball
	// ============================================================
	// Guide-equivalent implementation:
	// positive velocity toward the ball / CAR_MAX_SPEED.
	// Moving away from the ball is not punished.
	// ============================================================

	class SpeedTowardBallReward : public Reward {
	public:
		virtual float GetReward(
			const Player& player,
			const GameState& state,
			bool isFinal
		) override {
			Vec posDiff = state.ball.pos - player.pos;
			float distToBall = posDiff.Length();

			if (distToBall <= 0.001f)
				return 0.0f;

			Vec dirToBall = posDiff / distToBall;
			float speedTowardBall = dirToBall.Dot(player.vel);

			if (speedTowardBall > 0.0f)
				return speedTowardBall / CommonValues::CAR_MAX_SPEED;

			return 0.0f;
		}
	};


	// ============================================================
	// Touch Quality Reward
	// ============================================================
	//
	// Continuous 0 -> 10 touch-quality scale.
	//
	// A touch is evaluated using three independent properties:
	//
	//   1. Velocity change - how much the touch changed the ball's
	//      velocity vector (impact / redirection).
	//   2. Shot power      - how fast the ball is travelling after
	//      the touch (strength of the resulting shot/play).
	//   3. Height          - how high the ball reached relative to
	//      the Rocket League ceiling.
	//
	// The quality is a weighted average, then scaled to [0, 10].
	// Direction toward goal is deliberately handled by
	// VelocityBallToGoalReward instead of duplicating it here.
	//
	// Weighting:
	//   Velocity change : 35%
	//   Shot power      : 45%
	//   Height          : 20%
	//
	// This means a powerful, meaningful touch scores much higher
	// than a weak tap, while high/aerial touches receive additional
	// credit. A touch with no measurable impact/power/height gets 0.
	// ============================================================

	class TouchQualityReward : public Reward {
	public:
		virtual float GetReward(
			const Player& player,
			const GameState& state,
			bool isFinal
		) override {
			if (!player.ballTouchedStep)
				return 0.0f;

			// Rocket League ball speed is capped around 6000 uu/s.
			// Use the ball's physical speed range rather than the car's
			// max speed so powerful shots can actually reach the top end
			// of this reward scale.
			constexpr float BALL_MAX_SPEED = 6000.0f;

			// --------------------------------------------------------
			// 1. Velocity-change / impact quality
			// --------------------------------------------------------
			Vec deltaVelocity =
				state.ball.vel - previousBallVelocity;

			float velocityChange =
				deltaVelocity.Length() / BALL_MAX_SPEED;

			velocityChange = RS_MAX(
				0.0f,
				RS_MIN(velocityChange, 1.0f)
			);


			// --------------------------------------------------------
			// 2. Shot power
			// --------------------------------------------------------
			// Measures the resulting ball speed after the touch.
			float shotPower =
				state.ball.vel.Length() / BALL_MAX_SPEED;

			shotPower = RS_MAX(
				0.0f,
				RS_MIN(shotPower, 1.0f)
			);


			// --------------------------------------------------------
			// 3. Height quality
			// --------------------------------------------------------
			// Normalize the ball's height against the playable ceiling.
			// Squaring the value makes small height changes less valuable
			// while giving substantially more credit to genuinely high
			// touches/aerial plays.
			constexpr float CEILING_Z = CommonValues::CEILING_Z;

			float heightQuality =
				state.ball.pos.z / CEILING_Z;

			heightQuality = RS_MAX(
				0.0f,
				RS_MIN(heightQuality, 1.0f)
			);

			heightQuality *= heightQuality;


			// --------------------------------------------------------
			// Combine into a 0 -> 10 score
			// --------------------------------------------------------
			constexpr float VELOCITY_CHANGE_WEIGHT = 0.35f;
			constexpr float SHOT_POWER_WEIGHT = 0.45f;
			constexpr float HEIGHT_WEIGHT = 0.20f;

			float quality =
				velocityChange * VELOCITY_CHANGE_WEIGHT
				+ shotPower * SHOT_POWER_WEIGHT
				+ heightQuality * HEIGHT_WEIGHT;

			quality = RS_MAX(
				0.0f,
				RS_MIN(quality, 1.0f)
			);

			return quality * 10.0f;
		}

		virtual void Reset(const GameState& state) override {
			previousBallVelocity = state.ball.vel;
		}

		virtual void PreStep(const GameState& state) override {
			previousBallVelocity = state.ball.vel;
		}

	private:
		Vec previousBallVelocity = Vec(0, 0, 0);
	};


	// ============================================================
	// Total Energy Reward
	// ============================================================

	class TotalEnergyReward : public Reward {
	public:
		explicit TotalEnergyReward(float weight = 1.0f)
			: weight_(weight) {}

		virtual float GetReward(
			const Player& player,
			const GameState& state,
			bool isFinal
		) override {
			if (player.isDemoed)
				return 0.0f;

			constexpr float MASS = 180.0f;
			constexpr float GRAVITY = 650.0f;
			constexpr float CEILING_Z = CommonValues::CEILING_Z;
			constexpr float CAR_MAX_SPEED = CommonValues::CAR_MAX_SPEED;
			constexpr float JUMP_IMPULSE = 292.0f;
			constexpr float BOOST_ENERGY_PER_100 = 7.87e5f;

			const float max_energy =
				(MASS * GRAVITY * (CEILING_Z - 17.0f))
				+ (0.5f * MASS * (CAR_MAX_SPEED * CAR_MAX_SPEED))
				+ (BOOST_ENERGY_PER_100 * 100.0f)
				+ (0.8f * 0.5f * MASS * (JUMP_IMPULSE * JUMP_IMPULSE))
				+ (0.9f * 0.5f * MASS * (600.0f * 600.0f));

			float energy = 0.0f;

			energy +=
				1.1f * MASS * GRAVITY * player.pos.z;

			float velocity = player.vel.Length();

			energy +=
				0.5f * MASS * (velocity * velocity);

			energy +=
				BOOST_ENERGY_PER_100 * player.boost * 100.0f;

			if (player.hasJumped) {
				energy +=
					0.8f * 0.5f * MASS
					* (JUMP_IMPULSE * JUMP_IMPULSE);
			}

			if (player.hasFlipped) {
				float dodge_impulse =
					(velocity <= 1700.0f)
					? 500.0f + (velocity / 17.0f)
					: 600.0f - (velocity - 1700.0f);

				dodge_impulse =
					RS_MAX(
						dodge_impulse - 25.0f,
						0.0f
					);

				energy +=
					0.9f * 0.5f * MASS
					* (dodge_impulse * dodge_impulse);
			}

			float normalizedEnergy = energy / max_energy;

			return normalizedEnergy * weight_;
		}

	private:
		float weight_;
	};

}
